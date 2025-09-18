import * as dotenv from "dotenv";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatFireworks } from "@langchain/community/chat_models/fireworks";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";

import {
  createStructuredChatAgent,
  AgentExecutor,
  createToolCallingAgent,
} from "langchain/agents";

import { TavilySearchResults } from "@langchain/community/tools/tavily_search";

import readline, { createInterface } from "readline";
import { AIMessage, HumanMessage } from "@langchain/core/messages";

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { FakeEmbeddings } from "langchain/embeddings/fake";
import { createRetrieverTool } from "langchain/tools/retriever";

dotenv.config();

// // Declare the Model
const llm = new ChatFireworks({
  model: "accounts/fireworks/models/kimi-k2-instruct-0905",
  temperature: 0,
  apiKey: process.env.FIREWORKS_API_KEY,
});

// Prompt template must have "input" and "agent_scratchpad input variables"
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant"],
  ["placeholder", "{chat_history}"],
  ["human", "{input}"],
  ["placeholder", "{agent_scratchpad}"],
]);

// Instantiate Cheerio to fetch data from a website URL
const loader = new CheerioWebBaseLoader(
  "https://js.langchain.com/docs/concepts/chat_models/"
);
const docs = await loader.load();

//Instatntiate the splitter
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200,
  chunkOverlap: 20,
});

const splittedDocs = await splitter.splitDocuments(docs);

const embedding = new FakeEmbeddings();

// Instantiate the Vector store to save into the memory
const vectorStore = await MemoryVectorStore.fromDocuments(
  splittedDocs,
  embedding
);

// Now retrieve the Data from the vector store
const retriever = vectorStore.asRetriever({
  k: 2,
});

// Create a Retriever Tool
const retrieverTool = createRetrieverTool(retriever, {
  name: "custom_tool",

  description: "Use this tool when seraching information about Multimodality",
});

const tools = [new TavilySearchResults(), retrieverTool];

const agent = await createToolCallingAgent({
  llm,
  tools,
  prompt,
});

const agentExecutor = new AgentExecutor({
  agent,
  tools,
});

const rl = createInterface({
  input: process.stdin,
  output: process.stdout,
});

const chatHistory = [];

const askQuestion = () => {
  rl.question("User: ", async (input) => {
    if (input.toLowerCase() === "exit") {
      rl.close();
      return;
    }

    const result = await agentExecutor.invoke({
      input: input,
      chat_history: chatHistory,
    });

    console.log("Agent: ", result.output);

    chatHistory.push(new HumanMessage(input));
    chatHistory.push(new AIMessage(result.output));

    askQuestion();
  });
};

askQuestion();
