import fs from "fs";
import path from "path";
import { spawn, execSync } from "child_process";
import readline from "readline";
import dotenv from "dotenv";

dotenv.config();

// Swarm Context Log setup
const logFilePath = "swarm_context.log";
fs.writeFileSync(logFilePath, ""); // Truncate on start

// Debug Log setup (truncated on start if DEBUG is true)
if (process.env.DEBUG === "true") {
  fs.writeFileSync("orchestrator.log", "");
}

// Override console methods to write to orchestrator.log as well when DEBUG=true
const originalConsoleLog = console.log;
const originalConsoleError = console.error;

console.log = function (...args) {
  originalConsoleLog.apply(console, args);
  if (process.env.DEBUG === "true") {
    const timestamp = new Date().toISOString();
    const msg = args.map(arg => typeof arg === "object" ? JSON.stringify(arg) : arg).join(" ");
    fs.appendFileSync("orchestrator.log", `[${timestamp}] [CONSOLE] ${msg}\n`);
  }
};

console.error = function (...args) {
  originalConsoleError.apply(console, args);
  if (process.env.DEBUG === "true") {
    const timestamp = new Date().toISOString();
    const msg = args.map(arg => typeof arg === "object" ? JSON.stringify(arg) : arg).join(" ");
    fs.appendFileSync("orchestrator.log", `[${timestamp}] [CONSOLE_ERROR] ${msg}\n`);
  }
};

function debugLog(action, details = "") {
  if (process.env.DEBUG === "true") {
    const timestamp = new Date().toISOString();
    const formattedDetails = details ? ` | ${details}` : "";
    const logLine = `[${timestamp}] [${action}]${formattedDetails}\n`;
    fs.appendFileSync("orchestrator.log", logLine);
  }
}

// Memory caches for instructions/roles (loaded once at startup)
let globalRulesText = "";
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

let currentTicketTokens = {};

// Cache to prevent duplicate processing due to Notion API indexing latency
const recentlyProcessedTickets = new Map(); // ticketId -> timestamp

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

  // Log request to debugLog and swarm_context.log
  debugLog("LLM_REQUEST", `Model: ${model} (${provider}), Messages count: ${messages.length}`);
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
    debugLog("LLM_ERROR", `Model: ${model}, Error: ${errMsg}`);
    fs.appendFileSync(logFilePath, `[${new Date().toISOString()}] ERROR: ${errMsg}\n\n`);
    throw new Error(errMsg);
  }

  const data = await res.json();

  // Log response
  const responseEntry = `[${new Date().toISOString()}] RESPONSE FROM ${model} (${provider}):\n` + JSON.stringify(data, null, 2) + "\n\n";
  fs.appendFileSync(logFilePath, responseEntry);

  if (data.error) {
    debugLog("LLM_ERROR", `Model: ${model}, API Error: ${data.error.message || JSON.stringify(data.error)}`);
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

    if (!currentTicketTokens[model]) {
      currentTicketTokens[model] = { cacheMiss: 0, output: 0 };
    }
    currentTicketTokens[model].cacheMiss += miss;
    currentTicketTokens[model].output += output;
  }

  const responseMessage = data.choices[0].message;
  const tokensDetails = data.usage
    ? `Prompt Tokens: ${data.usage.prompt_tokens}, Completion: ${data.usage.completion_tokens}`
    : "No token usage details";
  debugLog("LLM_RESPONSE", `Model: ${model}, Content length: ${responseMessage.content ? responseMessage.content.length : 0} chars, Tool calls: ${responseMessage.tool_calls ? responseMessage.tool_calls.length : 0} | ${tokensDetails}`);

  return responseMessage;
}

// Test model connectivity on startup
async function verifyModels() {
  const roles = ["CEO", "CTO", "CCO", "CFO", "MODERATOR"];
  const provider = (process.env.LLM_PROVIDER || "OPENROUTER").toUpperCase();
  console.log(`[Старт] Перевірка конфігурації та доступності моделей (${provider})...`);
  debugLog("MODEL_VERIFICATION_START", `Provider: ${provider}`);

  for (const role of roles) {
    const modelVar = `${provider}_ROLE_${role}_MODEL`;
    const modelName = process.env[modelVar];
    if (!modelName) {
      console.error(`[Помилка] Змінна ${modelVar} не визначена у файлі .env.`);
      debugLog("MODEL_VERIFICATION_ERROR", `${modelVar} is undefined`);
      process.exit(1);
    }

    try {
      // 1-token test call
      await callLLM(modelName, [{ role: "user", content: "test" }], null, 1);
      console.log(`[Успішно] Модель для ролі ${role} (${modelName}) [${provider}] доступна.`);
      debugLog("MODEL_VERIFIED", `Role: ${role}, Model: ${modelName}`);
    } catch (err) {
      console.error(`[Помилка] Модель для ролі ${role} (${modelName}) [${provider}] недоступна або API-ключ недійсний. Деталі: ${err.message}`);
      debugLog("MODEL_VERIFICATION_FAILED", `Role: ${role}, Model: ${modelName}, Error: ${err.message}`);
      process.exit(1);
    }
  }
  console.log("[Старт] Усі моделі успішно верифіковано.");
  debugLog("MODEL_VERIFICATION_SUCCESS", "All models verified successfully.");
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

  debugLog("NOTION_WRITE_COMMENT_START", `Ticket: ${pageId}, Length: ${text.length} chars`);

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
    debugLog("NOTION_WRITE_COMMENT_SUCCESS", `Ticket: ${pageId}`);
  } catch (err) {
    console.error(`[Notion Error] Failed to append comment: ${err.message}`);
    debugLog("NOTION_WRITE_COMMENT_ERROR", `Ticket: ${pageId}, Error: ${err.message}`);
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
          } catch (e) { }
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
  debugLog("NOTION_SCAN_START", `Querying database ID: ${ticketsDbId}`);
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
    debugLog("NOTION_SCAN_ERROR", `Failed to parse response: ${JSON.stringify(result)}`);
    return [];
  }

  // Sort by update date (newest first)
  const results = data.results;
  results.sort((a, b) => new Date(b.last_edited_time) - new Date(a.last_edited_time));
  debugLog("NOTION_SCAN_END", `Found ${results.length} active tickets.`);
  return results;
}

// Helper to execute an expert agent (CEO/CTO/CCO/CFO) programmatically
async function executeAgent(role, prompt, ticket, description, comments) {
  const roleLower = role.toLowerCase();
  const provider = (process.env.LLM_PROVIDER || "OPENROUTER").toUpperCase();
  const roleModel = process.env[`${provider}_ROLE_${role.toUpperCase()}_MODEL`];
  const rolePromptSystem = rolePrompts[roleLower] || "";
  const limit = parseInt(process.env.TICKET_PROCESS_TOKENS_LIMIT, 10) || 50000;

  const currentDate = new Date().toLocaleDateString('uk-UA', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
  const agentEnvInfo = `\n\n[ІНФОРМАЦІЯ ПРО ПОТОЧНЕ СЕРЕДОВИЩЕ ШІ]:\n` +
    `- Поточна дата й час: ${new Date().toISOString()} (${currentDate})\n` +
    `- Провайдер LLM: ${provider}\n` +
    `- Модель, яка виконує роль ${role}: ${roleModel}\n` +
    `- Пошуковий інструмент: search_web (Google Search через google-surf-mcp).\n` +
    `- Важливо: Ти працюєш на моделі сімейства DeepSeek (${provider === "DEEPSEEK" ? "пряме підключення до api.deepseek.com" : "через проксі OpenRouter"}). Уникай будь-яких згадок про Gemini чи OpenAI як про твою поточну модель.`;

  const agentMessages = [
    { role: "system", content: globalRulesText + "\n\n" + rolePromptSystem + agentEnvInfo },
    {
      role: "user",
      content: `Контекст завдання (Тікет "${ticket.properties.Name?.title?.map(t => t.plain_text).join("") || "Без назви"}"):
Опис: ${description}
Попередні дебати/коментарі:
${comments.join("\n")}

Вказівка для виконання:
${prompt}`
    }
  ];

  debugLog("AGENT_INVOCATION_START", `Role: ${role}, Model: ${roleModel}`);
  let agentTurn = 1;
  const maxAgentTurns = 5;
  let agentProcessing = true;
  let finalAgentReply = "";

  const agentTools = [
    {
      type: "function",
      function: {
        name: "search_web",
        description: "Шукає інформацію в інтернеті за допомогою сервісу google-surf-mcp.",
        parameters: {
          type: "object",
          properties: { query: { type: "string", description: "Пошуковий запит." } },
          required: ["query"]
        }
      }
    },
    {
      type: "function",
      function: {
        name: "lexical_search",
        description: "Шукає ключові слова (через кому) у наданому тексті та повертає лише найважливіші речення або абзаци.",
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
        description: "Завантажує веб-сторінку за URL та витягує лише ті текстові сегменти, які містять вказані ключові слова (через кому).",
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
        description: "Пошук статей в базі знань Notion Wiki за властивістю 'Short Summary'. Повертає тільки ID, заголовок та короткий опис.",
        parameters: {
          type: "object",
          properties: { query: { type: "string", description: "Запит для пошуку статей Wiki за властивістю 'Short Summary'." } },
          required: ["query"]
        }
      }
    },
    {
      type: "function",
      function: {
        name: "read_wiki_page_body",
        description: "Зчитує повний вміст сторінки Wiki за її ID (використовувати тільки коли Short Summary на 100% відповідає потребі).",
        parameters: {
          type: "object",
          properties: { page_id: { type: "string", description: "ID сторінки Notion Wiki." } },
          required: ["page_id"]
        }
      }
    }
  ];

  while (agentProcessing && agentTurn <= maxAgentTurns) {
    const agentCurrentCost = Object.values(currentTicketTokens).reduce((sum, stats) => sum + stats.cacheMiss + stats.output, 0);
    if (agentCurrentCost > limit) {
      finalAgentReply = `⚠️ [Помилка] Виконання агента ${role} зупинено через перевищення ліміту токенів.`;
      debugLog("AGENT_TOKEN_LIMIT_EXCEEDED", `Role: ${role}, Cost: ${agentCurrentCost} > ${limit}`);
      break;
    }

    const response = await callLLM(roleModel, agentMessages, agentTools);
    if (response.tool_calls && response.tool_calls.length > 0) {
      agentMessages.push(response);
      for (const toolCall of response.tool_calls) {
        const { name, arguments: toolArgsString } = toolCall.function || toolCall;
        let toolArgs = {};
        try {
          toolArgs = JSON.parse(toolArgsString);
        } catch (e) {
          toolArgs = {};
        }

        console.log(`[Агент ${role}] Виклик інструменту: ${name} з аргументами: ${toolArgsString}`);
        debugLog("AGENT_TOOL_CALL", `Role: ${role}, Tool: ${name}, Args: ${toolArgsString}`);

        let toolResultText = "";
        try {
          if (name === "search_web") {
            const { query } = toolArgs;
            const mcpResponse = await googleMcp.callTool(googleMcp.searchToolName, { query });
            toolResultText = formatSearchResponse(mcpResponse);
          } else if (name === "lexical_search") {
            const { text, keywords } = toolArgs;
            const kwList = keywords.split(',').map(k => k.trim().toLowerCase()).filter(Boolean);
            const sentences = text.split(/(?:\. |\n)/);
            const matched = sentences.filter(s => {
              const lower = s.toLowerCase();
              return kwList.some(kw => lower.includes(kw));
            }).slice(0, 15);
            toolResultText = matched.length > 0
              ? JSON.stringify({ results: matched.map(s => s.trim()) })
              : JSON.stringify({ message: "Не знайдено збігів за ключовими словами." });
          } else if (name === "visit_page_lexical_search") {
            const { url, keywords } = toolArgs;
            const lexicalResult = await visitPageLexicalSearch(url, keywords);
            toolResultText = JSON.stringify(lexicalResult);
          } else if (name === "read_wiki") {
            const { query } = toolArgs;
            const wikiQuery = await notionMcp.callTool("API-query-data-source", {
              data_source_id: wikiDbId,
              filter: {
                property: "Short Summary",
                rich_text: { contains: query }
              }
            });
            const wikiData = parseMcpResponse(wikiQuery);
            if (wikiQuery.isError || (wikiData && (wikiData.object === "error" || wikiData.status >= 400))) {
              const errMsg = wikiData?.message || JSON.stringify(wikiData);
              toolResultText = `Помилка пошуку у Wiki: ${errMsg}`;
              debugLog("AGENT_WIKI_QUERY_ERROR", `Role: ${role}, Query: "${query}", Error: ${errMsg}`);
            } else if (!wikiData || !wikiData.results || wikiData.results.length === 0) {
              toolResultText = `Статей у Wiki з описом "${query}" не знайдено.`;
            } else {
              let wikiContent = "";
              for (const wikiPage of wikiData.results) {
                const title = wikiPage.properties.Name?.title?.map(t => t.plain_text).join("") || "Без назви";
                const summary = wikiPage.properties["Short Summary"]?.rich_text?.map(t => t.plain_text).join("") || "";
                wikiContent += `Wiki Page ID: "${wikiPage.id}"\nTitle: "${title}"\nShort Summary: "${summary}"\n\n`;
              }
              toolResultText = wikiContent;
            }
          } else if (name === "read_wiki_page_body") {
            const { page_id } = toolArgs;
            try {
              const body = await getPageDescription(page_id);
              toolResultText = body ? `Вміст сторінки:\n${body}` : "Вміст порожній або сторінку не знайдено.";
            } catch (err) {
              toolResultText = `Помилка читання сторінки: ${err.message}`;
            }
          } else {
            toolResultText = `Помилка: невідомий інструмент "${name}" для субагента.`;
          }
        } catch (toolErr) {
          console.error(`[Агент ${role}] Помилка інструменту ${name}:`, toolErr.message);
          debugLog("AGENT_TOOL_ERROR", `Role: ${role}, Tool: ${name}, Error: ${toolErr.message}`);
          toolResultText = `Помилка під час виконання інструменту: ${toolErr.message}`;
        }

        debugLog("AGENT_TOOL_RESULT", `Role: ${role}, Tool: ${name}, Result size: ${toolResultText.length} chars`);
        agentMessages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          name: name,
          content: toolResultText
        });
      }
    } else {
      finalAgentReply = response.content || "";
      agentProcessing = false;
    }
    agentTurn++;
  }

  debugLog("AGENT_INVOCATION_END", `Role: ${role}, Result size: ${finalAgentReply.length} chars`);
  return finalAgentReply;
}

// Helper to execute MODERATOR programmatically for QA and synthesis
async function runModerator(ticket, description, comments, agentAnswers) {
  const provider = (process.env.LLM_PROVIDER || "OPENROUTER").toUpperCase();
  const modelName = process.env[`${provider}_ROLE_MODERATOR_MODEL`];
  const promptSystem = rolePrompts.moderator || "";

  const currentDate = new Date().toLocaleDateString('uk-UA', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
  const moderatorEnvInfo = `\n\n[ІНФОРМАЦІЯ ПРО ПОТОЧНЕ СЕРЕДОВИЩЕ ШІ]:\n` +
    `- Поточна дата й час: ${new Date().toISOString()} (${currentDate})\n` +
    `- Провайдер LLM: ${provider}\n` +
    `- Модель, яка виконує роль MODERATOR: ${modelName}\n` +
    `- Важливо: Ти працюєш на reasoning моделі сімейства DeepSeek (${provider === "DEEPSEEK" ? "пряме підключення до api.deepseek.com" : "через проксі OpenRouter"}).`;

  let answersContent = "=== ВІДПОВІДІ ЕКСПЕРТНИХ АГЕНТІВ ===\n";
  for (const [role, answer] of Object.entries(agentAnswers)) {
    answersContent += `👤 [Агент ${role.toUpperCase()}]:\n${answer}\n\n`;
  }

  const messages = [
    { role: "system", content: globalRulesText + "\n\n" + promptSystem + moderatorEnvInfo },
    {
      role: "user",
      content: `Контекст завдання (Тікет "${ticket.properties.Name?.title?.map(t => t.plain_text).join("") || "Без назви"}"):
Опис: ${description}
Історія коментарів у Notion:
${comments.join("\n")}

${answersContent}
Будь ласка, виконай аудит відповідей, виріши спори та сформуй фінальний висновок для DENIS за правилами виводу.`
    }
  ];

  debugLog("MODERATOR_INVOCATION_START", `Model: ${modelName}`);
  const response = await callLLM(modelName, messages);
  debugLog("MODERATOR_INVOCATION_END", `Result size: ${response.content ? response.content.length : 0} chars`);
  return response.content || "";
}

// Processing individual tickets
async function processTicket(ticket) {
  const ticketId = ticket.id;
  if (recentlyProcessedTickets.has(ticketId)) {
    return;
  }
  recentlyProcessedTickets.set(ticketId, true);
  debugLog("TICKET_LOCK_ACQUIRED", `Ticket: ${ticketId}`);

  try {
    currentTicketTokens = {};

    const ticketTitle = ticket.properties.Name?.title?.map(t => t.plain_text).join("") || "Без назви";
    const ticketStatus = ticket.properties.Status?.status?.name || "Невідомо";
    const ticketAssigneeRole = ticket.properties["Assignee Role"]?.select?.name || "Невідомо";
    const ticketTokenMultiplier = ticket.properties["Token Multiplier Estimate"]?.rich_text?.map(t => t.plain_text).join("") || "";

    console.log(`[Активно] Обробка тікета: "${ticketTitle}" (${ticketAssigneeRole})`);
    debugLog("TICKET_PROCESS_START", `Ticket: "${ticketTitle}" (${ticketId}), Status: ${ticketStatus}, Assignee: ${ticketAssigneeRole}, Token Multiplier: ${ticketTokenMultiplier || "none"}`);

    let description = "";
    let comments = [];
    try {
      description = await getPageDescription(ticketId);
      comments = await getPageComments(ticketId);
    } catch (err) {
      console.error(`[Помилка] Не вдалося завантажити деталі тікета "${ticketTitle}":`, err.message);
      return;
    }

    let agentAnswers = {};
    let commentsHistory = [...comments];

    try {
      if (ticketAssigneeRole === "AI") {
        // Consensus flow - run CEO, CTO, CCO, CFO sequentially
        const rolesToCall = ["CEO", "CTO", "CCO", "CFO"];
        for (const role of rolesToCall) {
          const prompt = `Проаналізуй опис завдання та надай свій стислий відгук як ${role}.`;
          const answer = await executeAgent(role, prompt, ticket, description, commentsHistory);

          commentsHistory.push(`[Агент ${role}] [${new Date().toISOString()}]: ${answer}`);
          agentAnswers[role] = answer;
        }
      } else if (["CEO", "CTO", "CCO", "CFO"].includes(ticketAssigneeRole)) {
        // Single agent flow
        const role = ticketAssigneeRole;
        const prompt = `Виконай це завдання як ${role}.`;
        const answer = await executeAgent(role, prompt, ticket, description, commentsHistory);

        commentsHistory.push(`[Агент ${role}] [${new Date().toISOString()}]: ${answer}`);
        agentAnswers[role] = answer;
      } else {
        console.log(`[Попередження] Невідома роль ${ticketAssigneeRole}. Пропуск.`);
        return;
      }

      // Now run the Moderator verification loop (entirely in-memory and logged locally)
      let moderatorApproved = false;
      let moderatorCycle = 1;
      const maxModeratorCycles = 3;
      let finalModeratorContent = "";
      let summaryBoxContent = "";

      while (!moderatorApproved && moderatorCycle <= maxModeratorCycles) {
        console.log(`[Модератор] Запуск аудиту та синтезу (Цикл ${moderatorCycle}/${maxModeratorCycles})...`);
        const moderatorOutput = await runModerator(ticket, description, commentsHistory, agentAnswers);

        if (moderatorOutput.trim().startsWith("RETRY:")) {
          // Parse RETRY: <ROLE> - <reason>
          const match = moderatorOutput.match(/^RETRY:\s*([A-Z]{3,4})\s*-\s*(.*)$/i);
          if (match) {
            const retryRole = match[1].toUpperCase();
            const retryReason = match[2];

            console.log(`[Модератор] Запит на доопрацювання для ${retryRole}: ${retryReason}`);
            debugLog("MODERATOR_RETRY", `Cycle: ${moderatorCycle}, Role: ${retryRole}, Reason: ${retryReason}`);

            if (["CEO", "CTO", "CCO", "CFO"].includes(retryRole)) {
              // Rerun the agent with the moderator's reason
              const prompt = `Модератор надіслав запит на доопрацювання вашого аналізу:\n"${retryReason}"\nБудь ласка, онови свою відповідь з урахуванням цього зауваження.`;
              const updatedAnswer = await executeAgent(retryRole, prompt, ticket, description, commentsHistory);

              agentAnswers[retryRole] = updatedAnswer;
              commentsHistory.push(`[Агент ${retryRole} (Оновлено)] [${new Date().toISOString()}]: ${updatedAnswer}`);
            } else {
              console.warn(`[Попередження] Модератор запросив доопрацювання для непідтримуваної ролі: ${retryRole}`);
              finalModeratorContent = moderatorOutput;
              moderatorApproved = true;
            }
          } else {
            console.warn(`[Попередження] Невірний формат RETRY від Модератора: "${moderatorOutput}"`);
            finalModeratorContent = moderatorOutput;
            moderatorApproved = true;
          }
        } else {
          // Approved! Parse Final Conclusion and Summary Box
          finalModeratorContent = moderatorOutput;

          const summaryParts = moderatorOutput.split("=== SUMMARY BOX ===");
          if (summaryParts.length > 1) {
            finalModeratorContent = summaryParts[0].replace("### 🎯 ФІНАЛЬНИЙ ВИСНОВОК", "").trim();
            summaryBoxContent = summaryParts[1].trim();
          } else {
            finalModeratorContent = moderatorOutput.replace("### 🎯 ФІНАЛЬНИЙ ВИСНОВОК", "").trim();
          }

          moderatorApproved = true;
          debugLog("MODERATOR_APPROVED", `Cycle: ${moderatorCycle}`);
        }

        moderatorCycle++;
      }

      // Post final moderator conclusion to Notion comments
      try {
        await notionAppendComment(ticketId, `**Модератор:**\n${finalModeratorContent}`);
        console.log(`[Успішно] Фінальний висновок модератора додано до коментарів.`);
      } catch (commentErr) {
        console.error(`[Помилка] Не вдалося додати висновок модератора в коментарі:`, commentErr.message);
      }

      // Update Notion ticket page properties (Status = Needs Review, Assignee = DENIS)
      const properties = {
        Status: { status: { name: "Needs Review" } },
        "Assignee Role": { select: { name: "DENIS" } }
      };

      debugLog("NOTION_PATCH_PAGE_START", `Ticket: ${ticketId}, Properties: ${JSON.stringify(properties)}`);
      const updateResult = await notionMcp.callTool("API-patch-page", {
        page_id: ticketId,
        properties
      });
      const updateData = parseMcpResponse(updateResult);
      if (updateResult.isError || (updateData && (updateData.object === "error" || updateData.status >= 400))) {
        const errMsg = updateData?.message || JSON.stringify(updateData);
        console.error(`[Помилка] Не вдалося оновити статус тікета в Notion: ${errMsg}`);
        debugLog("TICKET_FINAL_UPDATE_ERROR", `Ticket: ${ticketId}, Error: ${errMsg}`);
      } else {
        console.log(`[Успішно] Тікет переведено в Needs Review (DENIS).`);
        debugLog("TICKET_FINAL_UPDATE_SUCCESS", `Ticket: ${ticketId}`);
      }

      // Append SUMMARY TOKENS
      let summaryTokensText = "";
      const isShort = process.env.TOKENS_SUMMARY_SHORT === "true";
      let totalTokensUsed = 0;

      if (isShort) {
        summaryTokensText = "=== SUMMARY TOKENS ===\n";
        for (const [mName, stats] of Object.entries(currentTicketTokens)) {
          const mTotal = stats.cacheMiss + stats.output;
          totalTokensUsed += mTotal;
          summaryTokensText += `- ${mName}: ${stats.cacheMiss} + ${stats.output} = ${mTotal}\n`;
        }
        summaryTokensText += `Total: ${totalTokensUsed}`;
      } else {
        summaryTokensText = "=== SUMMARY TOKENS ===\nЗагальна кількість витрачених токенів на обробку тікета (детально по моделям):\n";
        for (const [mName, stats] of Object.entries(currentTicketTokens)) {
          const mTotal = stats.cacheMiss + stats.output;
          totalTokensUsed += mTotal;
          summaryTokensText += `Модель: ${mName}\n- Input (Cache miss): ${stats.cacheMiss}\n- Output: ${stats.output}\n- Всього: ${mTotal}\n\n`;
        }
        summaryTokensText += `Загалом по всім моделям: ${totalTokensUsed} токенів`;
      }
      try {
        await notionAppendComment(ticketId, summaryTokensText);
        console.log(`[Токени] Загальний звіт додано до коментарів: ${totalTokensUsed} токенів.`);
      } catch (commentErr) {
        console.error(`[Помилка] Не вдалося додати SUMMARY TOKENS в коментарі:`, commentErr.message);
      }

    } catch (err) {
      console.error(`[Помилка обробки]`, err.message);
      debugLog("TICKET_PROCESS_ERROR", err.message);

      // Update Notion ticket status on processing error
      try {
        await notionAppendComment(ticketId, `⚠️ [Помилка обробки] Обробку тікета зупинено через критичну помилку: ${err.message}`);
        await notionMcp.callTool("API-patch-page", {
          page_id: ticketId,
          properties: {
            Status: { status: { name: "Needs Review" } },
            "Assignee Role": { select: { name: "DENIS" } }
          }
        });
        debugLog("TICKET_ERROR_STATUS_UPDATED", `Ticket: ${ticketId}, Status set to Needs Review, Assignee set to DENIS`);
      } catch (patchErr) {
        console.error(`[Помилка] Не вдалося оновити статус тікета після помилки:`, patchErr.message);
      }
    }
  } finally {
    recentlyProcessedTickets.delete(ticketId);
    debugLog("TICKET_LOCK_RELEASED", `Ticket: ${ticketId}`);
  }
}

// Main polling loop
let isProcessing = false;

async function checkAndProcess() {
  if (isProcessing) return;
  isProcessing = true;
  debugLog("POLL_START", "Scanning active tickets...");

  try {
    const tickets = await scanNotionTickets();
    const activeTickets = tickets.filter(t => {
      if (recentlyProcessedTickets.has(t.id)) {
        debugLog("POLL_LOCK_SKIP", `Skipping ticket ${t.id} ("${t.properties.Name?.title?.map(x => x.plain_text).join("")}") as it is currently being processed.`);
        return false;
      }
      return true;
    });

    if (activeTickets.length > 0) {
      console.log(`[Активно] Знайдено ${activeTickets.length} тікетів для обробки.`);
      debugLog("POLL_ACTIVE_TICKETS", `Processing ${activeTickets.length} tickets sequentially`);
      // Process sequentially to ensure correct sequence of logs and updates
      for (const ticket of activeTickets) {
        await processTicket(ticket);
      }
      console.log("[Чергування] Усі тікети поточної ітерації успішно оброблені.");
    } else {
      debugLog("POLL_NO_TICKETS", "No active tickets found");
    }
  } catch (err) {
    console.error("[Помилка циклу]", err.message);
    debugLog("POLL_ERROR", err.message);
  } finally {
    isProcessing = false;
    debugLog("POLL_END", "Polling cycle completed");
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

function terminatePreviousInstances() {
  const currentPid = process.pid;
  console.log(`[Ініціалізація] Поточний PID: ${currentPid}. Пошук застарілих процесів orchestrator.js...`);

  try {
    if (process.platform === "win32") {
      // Query node processes running orchestrator.js on Windows. We filter by name node.exe.
      const cmd = `powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter 'Name = \\"node.exe\\"' | Where-Object { $_.CommandLine -like \\"*orchestrator.js*\\" } | Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress"`;
      let output = "";
      try {
        output = execSync(cmd, { encoding: "utf-8" }).trim();
      } catch (err) {
        // May fail if no matching processes or powershell error
      }
      if (output) {
        let processes = [];
        try {
          const parsed = JSON.parse(output);
          processes = Array.isArray(parsed) ? parsed : [parsed];
        } catch (e) {
          // Parsing failed or empty
        }
        for (const proc of processes) {
          const pid = proc.ProcessId;
          if (pid && pid !== currentPid) {
            console.log(`[Ініціалізація] Завершення старого процесу orchestrator.js з PID ${pid}...`);
            try {
              process.kill(pid, "SIGKILL");
            } catch (err) {
              try { execSync(`taskkill /F /PID ${pid}`); } catch (kErr) { }
            }
          }
        }
      }
    } else {
      // Unix: ps aux
      let output = "";
      try {
        output = execSync("ps aux | grep node | grep orchestrator.js | grep -v grep", { encoding: "utf-8" }).trim();
      } catch (err) {
        // May fail if no processes found
      }
      if (output) {
        const lines = output.split("\n");
        for (const line of lines) {
          const parts = line.trim().split(/\s+/);
          const pid = parseInt(parts[1], 10);
          if (pid && pid !== currentPid) {
            console.log(`[Ініціалізація] Завершення старого процесу orchestrator.js з PID ${pid}...`);
            try {
              process.kill(pid, "SIGKILL");
            } catch (err) { }
          }
        }
      }
    }
  } catch (globalErr) {
    // Fail-safe
    console.error("[Ініціалізація] Помилка під час очищення попередніх процесів:", globalErr.message);
  }

  // Also write/update the PID file
  try {
    fs.writeFileSync("orchestrator.pid", currentPid.toString(), "utf-8");
  } catch (e) { }
}

async function debugTicketData(queryOrId) {
  console.log(`[Відлагодження] Отримання даних для тікету: "${queryOrId}"...`);
  let ticketId = queryOrId.trim();

  // Simple UUID regex test
  const isUuid = /^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?-?[0-9a-f]{12}$/i.test(ticketId);

  let ticket = null;
  if (isUuid) {
    try {
      const result = await notionMcp.callTool("API-retrieve-a-page", { page_id: ticketId });
      ticket = parseMcpResponse(result);
    } catch (err) {
      console.error(`[Помилка] Не вдалося завантажити тікет за ID ${ticketId}: ${err.message}`);
    }
  }

  // If not found by ID or not a UUID, query database by name
  if (!ticket) {
    try {
      const queryResult = await notionMcp.callTool("API-query-data-source", {
        data_source_id: ticketsDbId,
        filter: {
          property: "Name",
          title: { contains: ticketId }
        }
      });
      const data = parseMcpResponse(queryResult);
      if (data && data.results && data.results.length > 0) {
        ticket = data.results[0];
        ticketId = ticket.id;
        console.log(`[Відлагодження] Знайдено тікет за назвою. ID: ${ticketId}`);
      } else {
        console.error(`[Помилка] Тікет "${ticketId}" не знайдено за назвою або ID.`);
        return;
      }
    } catch (err) {
      console.error(`[Помилка] Збій під час пошуку тікета: ${err.message}`);
      return;
    }
  }

  // Fetch properties, description, comments
  try {
    const ticketTitle = ticket.properties.Name?.title?.map(t => t.plain_text).join("") || "Без назви";
    const ticketStatus = ticket.properties.Status?.status?.name || "Невідомо";
    const ticketAssigneeRole = ticket.properties["Assignee Role"]?.select?.name || "Невідомо";
    const ticketTokenMultiplier = ticket.properties["Token Multiplier Estimate"]?.rich_text?.map(t => t.plain_text).join("") || "";
    const description = await getPageDescription(ticketId);
    const comments = await getPageComments(ticketId);

    const debugOutput = {
      id: ticketId,
      title: ticketTitle,
      status: ticketStatus,
      assigneeRole: ticketAssigneeRole,
      tokenMultiplierEstimate: ticketTokenMultiplier,
      description,
      comments
    };

    console.log("\n=================== ДАНІ ТІКЕТУ ===================");
    console.log(`ID: ${debugOutput.id}`);
    console.log(`Назва: ${debugOutput.title}`);
    console.log(`Статус: ${debugOutput.status}`);
    console.log(`Виконавець: ${debugOutput.assigneeRole}`);
    console.log(`Оцінка токенів: ${debugOutput.tokenMultiplierEstimate}`);
    console.log("\n--- Опис (Опис сторінки) ---");
    console.log(debugOutput.description || "[Порожній опис]");
    console.log("\n--- Коментарі ---");
    if (debugOutput.comments.length > 0) {
      debugOutput.comments.forEach(c => console.log(c));
    } else {
      console.log("[Коментарі відсутні]");
    }
    console.log("===================================================\n");

    // Write to local json file for easy reading by AI assistant
    fs.writeFileSync("debug_ticket.json", JSON.stringify(debugOutput, null, 2), "utf-8");
    console.log("[Відлагодження] Повні дані тікету збережено у файл debug_ticket.json");

  } catch (err) {
    console.error(`[Помилка] Не вдалося зібрати дані тікету: ${err.message}`);
  }
}

// Start CLI Application
async function main() {
  terminatePreviousInstances();
  debugLog("STARTUP", "Initializing CLI Application...");
  // Load templates once
  try {
    globalRulesText = fs.readFileSync(".system_roles/GLOBAL_RULES.md", "utf-8");
    rolePrompts.ceo = fs.readFileSync(".system_roles/prompt_ceo.md", "utf-8");
    rolePrompts.cto = fs.readFileSync(".system_roles/prompt_cto.md", "utf-8");
    rolePrompts.cco = fs.readFileSync(".system_roles/prompt_cco.md", "utf-8");
    rolePrompts.cfo = fs.readFileSync(".system_roles/prompt_cfo.md", "utf-8");
    rolePrompts.moderator = fs.readFileSync(".system_roles/prompt_moderator.md", "utf-8");
    debugLog("TEMPLATES_LOADED", "Loaded GLOBAL_RULES.md, and all agent roles prompts");
  } catch (err) {
    console.error("[Помилка] Не вдалося зчитати інструкції або файли профілів ролей:", err.message);
    debugLog("STARTUP_ERROR", `Failed to read instructions or roles: ${err.message}`);
    process.exit(1);
  }

  // Check models
  await verifyModels();

  if (process.argv.includes("--check-only")) {
    console.log("[Завершено] Режим перевірки успішно пройдено.");
    debugLog("STARTUP_CHECK_ONLY", "Verification mode complete. Exiting.");
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
    } catch (e) { }

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
    debugLog("DATABASE_DISCOVERY", `Tickets DB: ${ticketsDbId}`);

    wikiDbId = await findDatabaseIdByName("wiki");
    console.log(`[Notion] Знайдено базу Wiki (wiki) ID: ${wikiDbId}`);
    debugLog("DATABASE_DISCOVERY", `Wiki DB: ${wikiDbId}`);
  } catch (err) {
    console.error("[Помилка] Не вдалося знайти необхідні бази даних у Notion:", err.message);
    debugLog("DATABASE_DISCOVERY_ERROR", err.message);
    notionMcp.stop();
    process.exit(1);
  }

  // CLI flag for ticket debugging
  if (process.argv.includes("--debug-ticket")) {
    const argIndex = process.argv.indexOf("--debug-ticket") + 1;
    const ticketQuery = process.argv[argIndex];
    if (!ticketQuery) {
      console.error("[Помилка] Вкажіть ID або назву тікета після прапорця --debug-ticket");
      notionMcp.stop();
      process.exit(1);
    }
    await debugTicketData(ticketQuery);
    notionMcp.stop();
    process.exit(0);
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
