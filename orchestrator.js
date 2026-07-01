import fs from "fs";
import path from "path";
import { spawn } from "child_process";
import readline from "readline";
import dotenv from "dotenv";

dotenv.config();

// Swarm Context Log setup
const logFilePath = "swarm_context.log";
fs.writeFileSync(logFilePath, ""); // Truncate on start

// Memory caches for instructions/roles (loaded once at startup)
let instructionsText = "";
const rolePrompts = {};

// MCP Client wrapper for subprocesses
class McpClient {
  constructor(name, command, args) {
    this.name = name;
    this.command = command;
    this.args = args;
    this.process = null;
    this.requestId = 1;
    this.pendingRequests = new Map();
    this.searchToolName = null;
  }

  async start() {
    console.log(`[MCP] Launching ${this.name}...`);
    const childEnv = { ...process.env };
    if (this.name === "Notion") {
      childEnv.NOTION_TOKEN = process.env.NOTION_API_KEY;
    }

    this.process = spawn(this.command, this.args, {
      shell: true,
      stdio: ["pipe", "pipe", "inherit"],
      env: childEnv
    });

    const rl = readline.createInterface({
      input: this.process.stdout,
      terminal: false
    });

    rl.on("line", (line) => {
      try {
        const response = JSON.parse(line);
        if (response.id !== undefined && this.pendingRequests.has(response.id)) {
          const { resolve, reject } = this.pendingRequests.get(response.id);
          this.pendingRequests.delete(response.id);
          if (response.error) {
            reject(new Error(response.error.message || JSON.stringify(response.error)));
          } else {
            resolve(response.result);
          }
        }
      } catch (err) {
        // Ignore non-JSON lines
      }
    });

    this.process.on("error", (err) => {
      console.error(`[MCP Error ${this.name}] Subprocess error:`, err.message);
    });

    // Initialize handshake
    await this.sendRequest("initialize", {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: { name: "PerspectAI", version: "1.0.0" }
    });

    this.sendNotification("notifications/initialized");
    console.log(`[MCP] ${this.name} initialized successfully.`);
  }

  sendRequest(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = this.requestId++;
      const payload = { jsonrpc: "2.0", id, method, params };
      this.pendingRequests.set(id, { resolve, reject });
      this.process.stdin.write(JSON.stringify(payload) + "\n");
    });
  }

  sendNotification(method, params = {}) {
    const payload = { jsonrpc: "2.0", method, params };
    this.process.stdin.write(JSON.stringify(payload) + "\n");
  }

  async callTool(name, argumentsObj) {
    return this.sendRequest("tools/call", {
      name,
      arguments: argumentsObj
    });
  }

  stop() {
    if (this.process) {
      this.process.kill();
    }
  }
}

// Global instances
let notionMcp;
let googleMcp;
let ticketsDbId = "";
let wikiDbId = "";

let currentTicketTokens = {
  cacheMiss: 0,
  output: 0
};

// Helper to parse MCP tool responses
function parseMcpResponse(mcpResult) {
  if (!mcpResult || !mcpResult.content || mcpResult.content.length === 0) {
    return null;
  }
  const textBlock = mcpResult.content.find(c => c.type === "text");
  if (!textBlock) return null;
  try {
    return JSON.parse(textBlock.text);
  } catch (err) {
    return textBlock.text;
  }
}

// Helper to call OpenRouter
// Helper to call the active LLM provider (OpenRouter or DeepSeek)
async function callLLM(model, messages, tools = null, maxTokens = null) {
  const provider = (process.env.LLM_PROVIDER || "OPENROUTER").toUpperCase();
  let apiKey = "";
  let baseUrl = "";
  let headers = {
    "Content-Type": "application/json"
  };

  if (provider === "DEEPSEEK") {
    apiKey = process.env.DEEPSEEK_API_KEY;
    baseUrl = "https://api.deepseek.com/chat/completions";
    headers["Authorization"] = `Bearer ${apiKey}`;
  } else {
    apiKey = process.env.OPENROUTER_API_KEY;
    baseUrl = "https://openrouter.ai/api/v1/chat/completions";
    headers["Authorization"] = `Bearer ${apiKey}`;
    headers["HTTP-Referer"] = "https://github.com/alenev/perspectAI";
    headers["X-Title"] = "PerspectAI Orchestrator";
  }

  const body = {
    model,
    messages,
    temperature: 0.7
  };
  if (tools) {
    body.tools = tools;
  }
  body.max_tokens = maxTokens || 4096;

  // Log request to swarm_context.log
  const logEntry = `[${new Date().toISOString()}] REQUEST TO ${model} (${provider}):\n` + JSON.stringify(body, null, 2) + "\n\n";
  fs.appendFileSync(logFilePath, logEntry);

  const res = await fetch(baseUrl, {
    method: "POST",
    headers,
    body: JSON.stringify(body)
  });

  if (!res.ok) {
    const errText = await res.text();
    const errMsg = `${provider} API error (HTTP ${res.status}): ${errText}`;
    fs.appendFileSync(logFilePath, `[${new Date().toISOString()}] ERROR: ${errMsg}\n\n`);
    throw new Error(errMsg);
  }

  const data = await res.json();

  // Log response
  const responseEntry = `[${new Date().toISOString()}] RESPONSE FROM ${model} (${provider}):\n` + JSON.stringify(data, null, 2) + "\n\n";
  fs.appendFileSync(logFilePath, responseEntry);

  if (data.error) {
    throw new Error(data.error.message || JSON.stringify(data.error));
  }

  if (data.usage) {
    const usage = data.usage;
    let hit = 0;
    if (typeof usage.prompt_cache_hit_tokens === "number") {
      hit = usage.prompt_cache_hit_tokens;
    } else if (usage.prompt_tokens_details && typeof usage.prompt_tokens_details.cached_tokens === "number") {
      hit = usage.prompt_tokens_details.cached_tokens;
    }
    const prompt = usage.prompt_tokens || 0;
    const miss = typeof usage.prompt_cache_miss_tokens === "number" 
      ? usage.prompt_cache_miss_tokens 
      : (prompt - hit);
    const output = usage.completion_tokens || 0;

    currentTicketTokens.cacheMiss += miss;
    currentTicketTokens.output += output;
  }

  return data.choices[0].message;
}

// Test model connectivity on startup
async function verifyModels() {
  const roles = ["ORCHESTRATOR", "CEO", "CTO", "CCO", "CFO"];
  const provider = (process.env.LLM_PROVIDER || "OPENROUTER").toUpperCase();
  console.log(`[Старт] Перевірка конфігурації та доступності моделей (${provider})...`);

  for (const role of roles) {
    const modelVar = `${provider}_ROLE_${role}_MODEL`;
    const modelName = process.env[modelVar];
    if (!modelName) {
      console.error(`[Помилка] Змінна ${modelVar} не визначена у файлі .env.`);
      process.exit(1);
    }

    try {
      // 1-token test call
      await callLLM(modelName, [{ role: "user", content: "test" }], null, 1);
      console.log(`[Успішно] Модель для ролі ${role} (${modelName}) [${provider}] доступна.`);
    } catch (err) {
      console.error(`[Помилка] Модель для ролі ${role} (${modelName}) [${provider}] недоступна або API-ключ недійсний. Деталі: ${err.message}`);
      process.exit(1);
    }
  }
  console.log("[Старт] Усі моделі успішно верифіковано.");
}

// Notion helper functions calling MCP tools
async function findDatabaseIdByName(name) {
  const result = await notionMcp.callTool("API-post-search", { query: name });
  const data = parseMcpResponse(result);
  if (!data || !data.results) {
    throw new Error(`Невірна відповідь пошуку від Notion MCP: ${JSON.stringify(result)}`);
  }
  const db = data.results.find(item => {
    if (item.object !== "database" && item.object !== "data_source") return false;
    const titleText = item.title ? item.title.map(t => t.plain_text).join("") : "";
    return titleText.toLowerCase() === name.toLowerCase();
  });
  if (!db) {
    throw new Error(`Базу знань/тікетів з назвою "${name}" не знайдено в Notion.`);
  }
  return db.id;
}

async function getPageDescription(pageId) {
  const result = await notionMcp.callTool("API-get-block-children", { block_id: pageId });
  const data = parseMcpResponse(result);
  if (!data || !data.results) return "";

  let text = "";
  for (const block of data.results) {
    if (block.type === "paragraph") {
      text += block.paragraph.rich_text.map(t => t.plain_text).join("") + "\n";
    } else if (block.type === "bulleted_list_item") {
      text += "- " + block.bulleted_list_item.rich_text.map(t => t.plain_text).join("") + "\n";
    } else if (block.type === "numbered_list_item") {
      text += "1. " + block.numbered_list_item.rich_text.map(t => t.plain_text).join("") + "\n";
    } else if (block.type === "heading_1") {
      text += "# " + block.heading_1.rich_text.map(t => t.plain_text).join("") + "\n";
    } else if (block.type === "heading_2") {
      text += "## " + block.heading_2.rich_text.map(t => t.plain_text).join("") + "\n";
    } else if (block.type === "heading_3") {
      text += "### " + block.heading_3.rich_text.map(t => t.plain_text).join("") + "\n";
    }
  }
  return text.trim();
}

async function getPageComments(pageId) {
  const result = await notionMcp.callTool("API-retrieve-a-comment", { block_id: pageId });
  const data = parseMcpResponse(result);
  if (!data || !data.results) return [];
  return data.results.map(c => {
    const author = c.created_by?.name || "User";
    const text = c.rich_text.map(t => t.plain_text).join("");
    return `[${author}] [${c.created_time}]: ${text}`;
  });
}

async function notionAppendComment(pageId, text) {
  if (!text || !text.trim()) return;
  const apiKey = process.env.NOTION_API_KEY;
  if (!apiKey) {
    console.error("[Notion Error] NOTION_API_KEY is not defined in env.");
    return;
  }
  
  try {
    const response = await fetch("https://api.notion.com/v1/comments", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        parent: { page_id: pageId },
        rich_text: [{
          type: "text",
          text: { content: text }
        }]
      })
    });
    
    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errText}`);
    }
  } catch (err) {
    console.error(`[Notion Error] Failed to append comment: ${err.message}`);
    throw err;
  }
}

async function notionAppendBlock(pageId, text) {
  try {
    await notionMcp.callTool("API-patch-block-children", {
      block_id: pageId,
      children: [
        {
          object: "block",
          type: "callout",
          callout: {
            rich_text: [
              {
                type: "text",
                text: { content: text }
              }
            ],
            icon: {
              type: "emoji",
              emoji: "🪙"
            },
            color: "gray_background"
          }
        }
      ]
    });
  } catch (err) {
    console.error(`[Notion Error] Failed to append block: ${err.message}`);
  }
}


function formatSearchResponse(mcpResponse) {
  if (!mcpResponse || !mcpResponse.content) return JSON.stringify(mcpResponse);
  const textBlock = mcpResponse.content.find(c => c.type === "text");
  if (!textBlock) return JSON.stringify(mcpResponse);
  try {
    const data = JSON.parse(textBlock.text);
    if (data.results && Array.isArray(data.results)) {
      const cleaned = data.results.slice(0, 5).map(r => ({
        title: r.title,
        url: r.url,
        snippet: r.description || r.snippet || ""
      }));
      return JSON.stringify({ results: cleaned });
    }
  } catch (e) {
    return textBlock.text;
  }
  return JSON.stringify(mcpResponse);
}

async function visitPageLexicalSearch(url, keywords) {
  try {
    const res = await fetch(url, {
      headers: {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
      }
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const html = await res.text();
    
    let text = html
      .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
      .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
      .replace(/<[^>]+>/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();

    const paragraphs = text.split(/(?:\. |\n)/);
    const kwList = keywords.split(',').map(k => k.trim().toLowerCase()).filter(Boolean);
    
    const matchingSegments = [];
    for (const para of paragraphs) {
      if (para.length < 15) continue;
      const lowerPara = para.toLowerCase();
      const matches = kwList.some(kw => lowerPara.includes(kw));
      if (matches) {
        matchingSegments.push(para.trim());
      }
      if (matchingSegments.length >= 10) break;
    }

    if (matchingSegments.length > 0) {
      return {
        url,
        segments: matchingSegments
      };
    }
    return { url, message: "Не знайдено текстових сегментів, що відповідають ключовим словам." };
  } catch (err) {
    return { url, error: `Не вдалося отримати або обробити сторінку: ${err.message}` };
  }
}



function compileDraftProgress(messages) {
  let draft = "### 📝 Чернетка проведеної роботи (до зупинки за лімітом токенів):\n\n";
  let hasContent = false;
  const agentCalls = {};

  for (const msg of messages) {
    if (msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        if (tc.function.name === "call_agent") {
          try {
            const args = JSON.parse(tc.function.arguments);
            agentCalls[tc.id] = args.role;
          } catch (e) {}
        }
      }
    }

    if (msg.role === "tool" && msg.name === "call_agent") {
      const role = agentCalls[msg.tool_call_id] || "Агент";
      draft += "👤 **Висновок " + role + ":**\n" + msg.content + "\n\n---\n\n";
      hasContent = true;
    }
  }

  return hasContent ? draft : "";
}

// Scans for active tickets via MCP
async function scanNotionTickets() {
  const statuses = (process.env.ACTIVE_STATUSES || "In progress")
    .split(",")
    .map(s => s.trim())
    .filter(Boolean);

  const roles = ["AI", "CEO", "CTO", "CCO", "CFO"];

  const statusFilter = {
    or: statuses.map(status => ({
      property: "Status",
      status: { equals: status }
    }))
  };

  const roleFilter = {
    or: roles.map(role => ({
      property: "Assignee Role",
      select: { equals: role }
    }))
  };

  const result = await notionMcp.callTool("API-query-data-source", {
    data_source_id: ticketsDbId,
    filter: { and: [statusFilter, roleFilter] }
  });

  const data = parseMcpResponse(result);
  if (!data || !data.results) {
    return [];
  }

  // Sort by update date (newest first)
  const results = data.results;
  results.sort((a, b) => new Date(b.last_edited_time) - new Date(a.last_edited_time));
  return results;
}

// Processing individual tickets
async function processTicket(ticket) {
  currentTicketTokens.cacheMiss = 0;
  currentTicketTokens.output = 0;

  const ticketId = ticket.id;
  const ticketTitle = ticket.properties.Name?.title?.map(t => t.plain_text).join("") || "Без назви";
  const ticketStatus = ticket.properties.Status?.status?.name || "Невідомо";
  const ticketAssigneeRole = ticket.properties["Assignee Role"]?.select?.name || "Невідомо";
  const ticketTokenMultiplier = ticket.properties["Token Multiplier Estimate"]?.rich_text?.map(t => t.plain_text).join("") || "";

  console.log(`[Активно] Обробка тікета: "${ticketTitle}" (${ticketAssigneeRole})`);

  let description = "";
  let comments = [];
  try {
    description = await getPageDescription(ticketId);
    comments = await getPageComments(ticketId);
  } catch (err) {
    console.error(`[Помилка] Не вдалося завантажити деталі тікета "${ticketTitle}":`, err.message);
    return;
  }

  // Main agent prompt setup
  const provider = (process.env.LLM_PROVIDER || "OPENROUTER").toUpperCase();
  const orchestratorModel = process.env[`${provider}_ROLE_ORCHESTRATOR_MODEL`];
  const currentDate = new Date().toLocaleDateString('uk-UA', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
  const envInfo = `\n\n[ІНФОРМАЦІЯ ПРО ПОТОЧНЕ СЕРЕДОВИЩЕ ШІ (Контекст виконання)]:\n` +
    `- Поточна дата й час: ${new Date().toISOString()} (${currentDate})\n` +
    `- Активний провайдер LLM: ${provider}\n` +
    `- Модель, яка виконує роль Оркестратора: ${orchestratorModel}\n` +
    `- Пошуковий сервіс для інтернету: search_web (Google Search через google-surf-mcp). Робота повністю залежить від доступності цього інструменту.\n\n` +
    `При веденні консиліуму та формуванні відповідей CTO/CEO/CCO/CFO обов'язково враховуйте, що робочими моделями є сімейство DeepSeek (${provider === "DEEPSEEK" ? "прямий API deepseek.com" : "через OpenRouter"}), а не Gemini чи OpenAI. Інформуйте CTO, щоб він чітко вказував ці моделі та інструменти при наданні фідбеку на технічні запити.`;

  const messages = [
    { role: "system", content: instructionsText + envInfo },
    {
      role: "user",
      content: `Виконайте роботу над наступним тікетом згідно з інструкцією INSTRUCTIONS.md:
Назва тікета: "${ticketTitle}"
ID тікета: ${ticketId}
Поточний статус: ${ticketStatus}
Роль виконавця (Assignee Role): ${ticketAssigneeRole}
Token Multiplier Estimate: ${ticketTokenMultiplier || "Не визначено"}

Опис завдання:
${description || "Опис відсутній"}

Історія коментарів у Notion:
${comments.length > 0 ? comments.join("\n") : "Коментарів немає."}

Будь ласка, координуйте дії рою, викликайте експертів (CEO/CTO/CCO/CFO) за допомогою інструменту call_agent, додавайте коментарі та оновлюйте властивості тікета.`
    }
  ];

  const tools = [
    {
      type: "function",
      function: {
        name: "call_agent",
        description: "Викликає конкретного агента (CEO, CTO, CCO, CFO) з промптом та повертає його коментар. Агент отримає опис тікета та історію дебатів.",
        parameters: {
          type: "object",
          properties: {
            role: { type: "string", enum: ["CEO", "CTO", "CCO", "CFO"], description: "Роль агента." },
            prompt: { type: "string", description: "Запитання або вказівка для агента." }
          },
          required: ["role", "prompt"]
        }
      }
    },
    {
      type: "function",
      function: {
        name: "notion_append_comment",
        description: "Додає новий коментар до поточного тікета в Notion.",
        parameters: {
          type: "object",
          properties: {
            text: { type: "string", description: "Текст коментаря." }
          },
          required: ["text"]
        }
      }
    },
    {
      type: "function",
      function: {
        name: "notion_update_ticket",
        description: "Оновлює властивості тікета в Notion (Status, Assignee Role, Token Multiplier Estimate, Summary Box). Якщо властивість 'Summary Box' відсутня, ви можете передати її в аргументі 'summaryBox' і скрипт запише її в коментар або тіло сторінки.",
        parameters: {
          type: "object",
          properties: {
            status: { type: "string", description: "Новий статус (наприклад: 'In progress', 'Needs Review', 'Done', 'Token Planning')." },
            assigneeRole: { type: "string", description: "Нова роль виконавця (наприклад: 'DENIS', 'AI', 'CEO', 'CTO', 'CCO', 'CFO')." },
            tokenMultiplierEstimate: { type: "string", description: "Оцінка токенів (наприклад: 'Потрібно x2 ліміту')." },
            summaryBox: { type: "string", description: "Резюме конфлікту (до 3 абзаців) для 'Summary Box'." }
          }
        }
      }
    },
    {
      type: "function",
      function: {
        name: "search_web",
        description: "Шукає інформацію в інтернеті за допомогою сервісу google-surf-mcp.",
        parameters: {
          type: "object",
          properties: {
            query: { type: "string", description: "Пошуковий запит." }
          },
          required: ["query"]
        }
      }
    },
    {
      type: "function",
      function: {
        name: "lexical_search",
        description: "Шукає ключові слова (через кому) у наданому тексті та повертає лише найважливіші речення або абзаци, щоб уникнути надмірності.",
        parameters: {
          type: "object",
          properties: {
            text: { type: "string", description: "Початковий текст для фільтрації." },
            keywords: { type: "string", description: "Ключові слова через кому для пошуку." }
          },
          required: ["text", "keywords"]
        }
      }
    },
    {
      type: "function",
      function: {
        name: "visit_page_lexical_search",
        description: "Завантажує веб-сторінку за URL та витягує лише ті текстові сегменти, які містять вказані ключові слова (через кому). Вкрай економить токени.",
        parameters: {
          type: "object",
          properties: {
            url: { type: "string", description: "URL веб-сторінки для аналізу." },
            keywords: { type: "string", description: "Ключові слова через кому." }
          },
          required: ["url", "keywords"]
        }
      }
    },
    {
      type: "function",
      function: {
        name: "read_wiki",
        description: "Пошук та читання статей в базі знань Notion Wiki за властивістю 'Short Summary'.",
        parameters: {
          type: "object",
          properties: {
            query: { type: "string", description: "Запит для пошуку статей Wiki за властивістю 'Short Summary'." }
          },
          required: ["query"]
        }
      }
    },
    {
      type: "function",
      function: {
        name: "update_roadmap",
        description: "Додає інформацію про завершене завдання у локальний файл ROADMAP.md.",
        parameters: {
          type: "object",
          properties: {
            content: { type: "string", description: "Текст для запису в ROADMAP.md." }
          },
          required: ["content"]
        }
      }
    }
  ];

  let turn = 1;
  const maxTurns = 15;
  let processing = true;

  while (processing && turn <= maxTurns) {
    try {
      const limit = parseInt(process.env.TICKET_PROCESS_TOKENS_LIMIT, 10) || 50000;
      const currentCost = currentTicketTokens.cacheMiss + currentTicketTokens.output;
      if (currentCost > limit) {
        console.log(`[Token Limit] Exceeded token limit for ticket: ${currentCost} (Limit: ${limit}). Stopping execution.`);
        const draftContent = compileDraftProgress(messages);
        if (draftContent) {
          await notionAppendComment(ticketId, draftContent);
        }
        await notionAppendComment(ticketId, `⚠️ Обробку зупинено: перевищено ліміт токенів для одного тікета (${currentCost} > ${limit} токенів Input (Cache miss)+Output).`);
        await notionMcp.callTool("API-patch-page", {
          page_id: ticketId,
          properties: {
            Status: { status: { name: "Token Planning" } },
            "Assignee Role": { select: { name: "DENIS" } },
            "Token Multiplier Estimate": {
              rich_text: [{ type: "text", text: { content: `Перевищено ліміт: ${currentCost} токенів` } }]
            }
          }
        });
        processing = false;
        break;
      }

      const provider = (process.env.LLM_PROVIDER || "OPENROUTER").toUpperCase();
      const modelName = process.env[`${provider}_ROLE_ORCHESTRATOR_MODEL`];
      const response = await callLLM(modelName, messages, tools);

      if (response.tool_calls && response.tool_calls.length > 0) {
        messages.push(response);

        for (const toolCall of response.tool_calls) {
          const { name, arguments: argsString } = toolCall.function;
          const args = JSON.parse(argsString);
          let resultText = "";

          console.log(`[Виклик інструменту] ${name} з аргументами: ${argsString}`);

          try {
            if (name === "call_agent") {
              const { role, prompt } = args;
              const limit = parseInt(process.env.TICKET_PROCESS_TOKENS_LIMIT, 10) || 50000;
              if (currentTicketTokens.cacheMiss + currentTicketTokens.output > limit) {
                resultText = `Помилка: Перевищено ліміт токенів для цього тікета (${currentTicketTokens.cacheMiss + currentTicketTokens.output} > ${limit}).`;
              } else {
                const roleModel = process.env[`${provider}_ROLE_${role.toUpperCase()}_MODEL`];
                const rolePromptSystem = rolePrompts[role.toLowerCase()] || "";

                const currentDate = new Date().toLocaleDateString('uk-UA', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
                const agentEnvInfo = `\n\n[ІНФОРМАЦІЯ ПРО ПОТОЧНЕ СЕРЕДОВИЩЕ ШІ]:\n` +
                   `- Поточна дата й час: ${new Date().toISOString()} (${currentDate})\n` +
                   `- Провайдер LLM: ${provider}\n` +
                   `- Модель, яка виконує роль ${role}: ${roleModel}\n` +
                   `- Пошуковий інструмент: search_web (Google Search через google-surf-mcp).\n` +
                   `- Важливо: Ти працюєш на моделі сімейства DeepSeek (${provider === "DEEPSEEK" ? "пряме підключення до api.deepseek.com" : "через проксі OpenRouter"}). Уникай будь-яких згадок про Gemini чи OpenAI як про твою поточну модель.`;

                const agentMessages = [
                  { role: "system", content: rolePromptSystem + agentEnvInfo },
                  {
                    role: "user",
                    content: `Контекст завдання (Тікет "${ticketTitle}"):
Опис: ${description}
Попередні дебати/коментарі:
${comments.join("\n")}

Запит оркестратора:
${prompt}`
                  }
                ];

                const reply = await callLLM(roleModel, agentMessages);
                resultText = reply.content || "";
              }
            } else if (name === "notion_append_comment") {
              const { text } = args;
              await notionAppendComment(ticketId, text);
              resultText = "Коментар успішно додано в Notion.";
            } else if (name === "notion_update_ticket") {
              const { status, assigneeRole, tokenMultiplierEstimate, summaryBox } = args;
              const properties = {};

              if (status) {
                properties.Status = { status: { name: status } };
              }
              if (assigneeRole) {
                properties["Assignee Role"] = { select: { name: assigneeRole } };
              }
              if (tokenMultiplierEstimate) {
                properties["Token Multiplier Estimate"] = {
                  rich_text: [{ type: "text", text: { content: tokenMultiplierEstimate } }]
                };
              }

              // Check if "Summary Box" or "Summary" exists in page properties, otherwise write as callout/comment
              if (summaryBox) {
                const summaryPropName = ticket.properties["Summary Box"] ? "Summary Box" : (ticket.properties["Summary"] ? "Summary" : null);
                if (summaryPropName) {
                  properties[summaryPropName] = {
                    rich_text: [{ type: "text", text: { content: summaryBox } }]
                  };
                } else {
                  // Append summaryBox as a comment
                  await notionAppendComment(ticketId, `=== SUMMARY BOX ===\n${summaryBox}`);
                }
              }

              await notionMcp.callTool("API-patch-page", {
                page_id: ticketId,
                properties
              });
              resultText = "Властивості тікета оновлено в Notion.";
            } else if (name === "search_web") {
              const { query } = args;
              try {
                const mcpResponse = await googleMcp.callTool(googleMcp.searchToolName, { query });
                resultText = formatSearchResponse(mcpResponse);
              } catch (err) {
                console.error(`[Критична помилка] Збій пошуку під час роботи: ${err.message}`);
                await reportSearchProblem(`Збій пошуку під час виконання запиту "${query}": ${err.message}`);
              }
            } else if (name === "lexical_search") {
              const { text, keywords } = args;
              const kwList = keywords.split(',').map(k => k.trim().toLowerCase()).filter(Boolean);
              const sentences = text.split(/(?:\. |\n)/);
              const matched = sentences.filter(s => {
                const lower = s.toLowerCase();
                return kwList.some(kw => lower.includes(kw));
              }).slice(0, 15);
              resultText = matched.length > 0 
                ? JSON.stringify({ results: matched.map(s => s.trim()) }) 
                : JSON.stringify({ message: "Не знайдено збігів за ключовими словами." });
            } else if (name === "visit_page_lexical_search") {
              const { url, keywords } = args;
              const lexicalResult = await visitPageLexicalSearch(url, keywords);
              resultText = JSON.stringify(lexicalResult);
            } else if (name === "read_wiki") {
              const { query } = args;
              // Query Wiki Database via MCP
              const wikiQuery = await notionMcp.callTool("API-query-data-source", {
                data_source_id: wikiDbId,
                filter: {
                  property: "Short Summary",
                  rich_text: { contains: query }
                }
              });

              const wikiData = parseMcpResponse(wikiQuery);
              if (!wikiData || !wikiData.results || wikiData.results.length === 0) {
                resultText = `Статей у Wiki з описом "${query}" не знайдено.`;
              } else {
                let wikiContent = "";
                for (const wikiPage of wikiData.results) {
                  const title = wikiPage.properties.Name?.title?.map(t => t.plain_text).join("") || "Без назви";
                  const summary = wikiPage.properties["Short Summary"]?.rich_text?.map(t => t.plain_text).join("") || "";
                  const body = await getPageDescription(wikiPage.id);
                  wikiContent += `Wiki Page: "${title}"\nSummary: ${summary}\nContent:\n${body}\n\n`;
                }
                resultText = wikiContent;
              }
            } else if (name === "update_roadmap") {
              const { content } = args;
              fs.appendFileSync("ROADMAP.md", content + "\n");
              resultText = "Локальний файл ROADMAP.md оновлено.";
            } else {
              resultText = `Помилка: невідомий інструмент "${name}".`;
            }
          } catch (toolErr) {
            console.error(`[Помилка інструменту ${name}]`, toolErr.message);
            resultText = `Помилка під час виконання інструменту: ${toolErr.message}`;
          }

          messages.push({
            role: "tool",
            tool_call_id: toolCall.id,
            name: name,
            content: resultText
          });
        }
      } else {
        // Complete processing, no more tools
        console.log(`[Успішно] Оркестратор завершив обробку тікета: "${ticketTitle}"`);
        console.log(`Відповідь: ${response.content}`);
        processing = false;
        
        // Append SUMMARY TOKENS directly here
        const totalTokensUsed = currentTicketTokens.cacheMiss + currentTicketTokens.output;
        const summaryTokensText = `=== SUMMARY TOKENS ===\nЗагальна кількість витрачених токенів на обробку тікета:\n- Input (Cache miss): ${currentTicketTokens.cacheMiss}\n- Output: ${currentTicketTokens.output}\n- Всього (Input miss + Output): ${totalTokensUsed} токенів`;
        try {
          await notionAppendComment(ticketId, summaryTokensText);
          console.log(`[Токени] Загальний звіт додано до коментарів: ${totalTokensUsed} токенів.`);
        } catch (commentErr) {
          console.error(`[Помилка] Не вдалося додати SUMMARY TOKENS в коментарі:`, commentErr.message);
        }
      }
    } catch (err) {
      console.error(`[Помилка обробки]`, err.message);
      processing = false;
    }
    turn++;
  }


}

// Main polling loop
let isProcessing = false;

async function checkAndProcess() {
  if (isProcessing) return;
  isProcessing = true;

  try {
    const tickets = await scanNotionTickets();
    if (tickets.length > 0) {
      console.log(`[Активно] Знайдено ${tickets.length} тікетів для обробки.`);
      // Process sequentially to ensure correct sequence of logs and updates
      for (const ticket of tickets) {
        await processTicket(ticket);
      }
      console.log("[Чергування] Усі тікети поточної ітерації успішно оброблені.");
    }
  } catch (err) {
    console.error("[Помилка циклу]", err.message);
  } finally {
    isProcessing = false;
  }
}

async function reportSearchProblem(errorMessage) {
  console.error(`[Критична помилка] Інструмент пошуку google-surf-mcp недоступний: ${errorMessage}`);
  
  // 1. Initialize Notion MCP if not already done
  if (!notionMcp) {
    notionMcp = new McpClient(
      "Notion",
      "npx",
      ["-y", "@notionhq/notion-mcp-server"]
    );
    try {
      await notionMcp.start();
    } catch (notionErr) {
      console.error("[Помилка] Не вдалося ініціалізувати Notion для звіту про помилку:", notionErr.message);
      process.exit(1);
    }
  }

  // 2. Discover ticketsDbId if not already done
  if (!ticketsDbId) {
    try {
      ticketsDbId = await findDatabaseIdByName("AI-Integrator Hub");
    } catch (notionErr) {
      console.error("[Помилка] Не вдалося знайти базу AI-Integrator Hub для створення тікета про помилку:", notionErr.message);
      process.exit(1);
    }
  }

  // 3. Create a ticket in Notion
  try {
    console.log("[Notion] Створення тікета про проблему з google-surf-mcp...");
    await notionMcp.callTool("API-post-page", {
      parent: { database_id: ticketsDbId },
      properties: {
        Name: {
          title: [{ type: "text", text: { content: "Проблема google-surf-mcp" } }]
        },
        "Assignee Role": {
          select: { name: "DENIS" }
        },
        Status: {
          status: { name: "Needs Review" }
        }
      },
      children: [
        {
          type: "paragraph",
          paragraph: {
            rich_text: [
              {
                type: "text",
                text: { content: `Інструмент пошуку google-surf-mcp недоступний або повернув помилку під час перевірки запуску:\n\nДеталі помилки:\n${errorMessage}` }
              }
            ]
          }
        }
      ]
    });
    console.log("[Notion] Тікет 'Проблема google-surf-mcp' успішно створено.");
  } catch (createErr) {
    console.error("[Помилка] Не вдалося створити тікет у Notion:", createErr.message);
  }

  // Stop everything and exit
  notionMcp.stop();
  if (googleMcp) googleMcp.stop();
  process.exit(1);
}

// Start CLI Application
async function main() {
  // Load templates once
  try {
    instructionsText = fs.readFileSync("INSTRUCTIONS.md", "utf-8");
    rolePrompts.ceo = fs.readFileSync(".system_roles/prompt_ceo.md", "utf-8");
    rolePrompts.cto = fs.readFileSync(".system_roles/prompt_cto.md", "utf-8");
    rolePrompts.cco = fs.readFileSync(".system_roles/prompt_cco.md", "utf-8");
    rolePrompts.cfo = fs.readFileSync(".system_roles/prompt_cfo.md", "utf-8");
  } catch (err) {
    console.error("[Помилка] Не вдалося зчитати інструкції або файли профілів ролей:", err.message);
    process.exit(1);
  }

  // Check models
  await verifyModels();

  if (process.argv.includes("--check-only")) {
    console.log("[Завершено] Режим перевірки успішно пройдено.");
    process.exit(0);
  }

  if (process.argv.includes("--search")) {
    const queryIndex = process.argv.indexOf("--lexical") !== -1 
      ? process.argv.indexOf("--lexical") + 1 
      : process.argv.indexOf("--search") + 1;
    const query = process.argv[queryIndex] || "погода в Україні";
    
    console.log(`[CLI Пошук] Запит: "${query}"`);
    
    googleMcp = new McpClient("GoogleSearch", "npx", ["-y", "google-surf-mcp"]);
    await googleMcp.start();
    
    const toolsList = await googleMcp.sendRequest("tools/list");
    const searchTool = toolsList.tools.find(t => t.name.toLowerCase().includes("search"));
    googleMcp.searchToolName = searchTool ? searchTool.name : "search";
    
    console.log(`[CLI Пошук] Виконання пошуку через ${googleMcp.searchToolName}...`);
    const mcpResponse = await googleMcp.callTool(googleMcp.searchToolName, { query });
    
    const fullText = JSON.stringify(mcpResponse, null, 2);
    
    let firstUrl = "";
    let filteredText = "";
    try {
      const textBlock = mcpResponse.content.find(c => c.type === "text");
      const parsed = JSON.parse(textBlock.text);
      if (parsed.results && parsed.results.length > 0) {
        firstUrl = parsed.results[0].url;
      }
    } catch (e) {}

    if (firstUrl) {
      console.log(`[CLI Пошук] Тестування visitPageLexicalSearch на URL: ${firstUrl}`);
      const lexicalResult = await visitPageLexicalSearch(firstUrl, "погода,температура,вітер,опади,курс,USD,гривня");
      filteredText = JSON.stringify(lexicalResult, null, 2);
    } else {
      filteredText = "Не знайдено URL для лексичного пошуку.";
    }

    const output = `=== a) ПОВНИЙ КОНТЕКСТ РЕЗУЛЬТАТУ ПОШУКУ ===\n${fullText}\n\n=== б) ВІДФІЛЬТРОВАНИЙ МЕТОДОМ visitPageLexicalSearch() РЕЗУЛЬТАТ ===\n${filteredText}\n`;
    fs.writeFileSync("search_result.txt", output, "utf8");
    console.log("[CLI Пошук] Результати успішно записано у файл search_result.txt");
    
    googleMcp.stop();
    process.exit(0);
  }

  // Initialize Notion MCP Client Subprocess
  notionMcp = new McpClient(
    "Notion",
    "npx",
    ["-y", "@notionhq/notion-mcp-server"]
  );
  try {
    await notionMcp.start();
  } catch (err) {
    console.error("[Помилка] Не вдалося запустити @notionhq/notion-mcp-server:", err.message);
    process.exit(1);
  }

  // Dynamically find Databases by name via MCP (so we have ticketsDbId before testing search)
  try {
    console.log("[Notion] Пошук баз даних за назвами...");
    ticketsDbId = await findDatabaseIdByName("AI-Integrator Hub");
    console.log(`[Notion] Знайдено базу тікетів (AI-Integrator Hub) ID: ${ticketsDbId}`);

    wikiDbId = await findDatabaseIdByName("wiki");
    console.log(`[Notion] Знайдено базу Wiki (wiki) ID: ${wikiDbId}`);
  } catch (err) {
    console.error("[Помилка] Не вдалося знайти необхідні бази даних у Notion:", err.message);
    notionMcp.stop();
    process.exit(1);
  }

  // Initialize Google Search MCP Client Subprocess
  googleMcp = new McpClient(
    "GoogleSearch",
    "npx",
    ["-y", "google-surf-mcp"]
  );
  try {
    await googleMcp.start();
    // Discover the search tool name
    try {
      const toolsList = await googleMcp.sendRequest("tools/list");
      const searchTool = toolsList.tools.find(t => t.name.toLowerCase().includes("search"));
      if (searchTool) {
        googleMcp.searchToolName = searchTool.name;
        console.log(`[MCP] Discovered search tool: "${googleMcp.searchToolName}"`);
      } else {
        googleMcp.searchToolName = "search";
        console.log(`[MCP] No search tool found in list, using fallback: "${googleMcp.searchToolName}"`);
      }
    } catch (err) {
      googleMcp.searchToolName = "search";
      console.log(`[MCP] Error listing tools, using default fallback: "${googleMcp.searchToolName}"`);
    }

    // Perform startup test search to verify google-surf-mcp is working
    console.log("[MCP] Тестування працездатності google-surf-mcp...");
    try {
      const testSearch = await googleMcp.callTool(googleMcp.searchToolName, { query: "test" });
      if (testSearch && testSearch.isError) {
        throw new Error(JSON.stringify(testSearch));
      }
      console.log("[MCP] Інструмент google-surf-mcp успішно протестовано.");
    } catch (testErr) {
      await reportSearchProblem(testErr.message);
    }
  } catch (err) {
    await reportSearchProblem(err.message);
  }



  // Polling loop
  const pollingInterval = parseInt(process.env.POLLING_INTERVAL_MS, 10) || 10000;
  console.log(`[Чергування] Запуск постійного циклу опитування Notion кожні ${pollingInterval / 1000} секунд.`);

  // First execution
  await checkAndProcess();

  // Run scheduler
  setInterval(checkAndProcess, pollingInterval);

  // Clean shutdown
  process.on("SIGINT", () => {
    console.log("[Чергування] Зупинка оркестратора...");
    notionMcp.stop();
    googleMcp.stop();
    process.exit(0);
  });
}

main();
