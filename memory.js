import * as dotenv from "dotenv";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { BufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";
import { UpstashRedisChatMessageHistory } from "@langchain/community/stores/message/upstash_redis";
import { RunnableSequence } from "@langchain/core/runnables";
dotenv.config();

const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash",
  temperature: 0.7,
  apiKey: process.env.GEMINI_API_KEY,
});

const prompt = ChatPromptTemplate.fromTemplate(
  `You are an AI Assistant
  History: {history}
  {input}`
);

const upstashChatHistory = new UpstashRedisChatMessageHistory({
  sessionId: "chat1",
  config: {
    url: process.env.UPSTASH_REDIS_REST_URL,
    token: process.env.UPSTASH_REDIS_REST_TOKEN,
  },
});

const memory = new BufferMemory({
  memoryKey: "history", // this key will help to append messages from the conversation and will inject into a variable presented in prompt.
  chatHistory: upstashChatHistory,
});

// Usage of Chain Classes to attach memory to prompt and model
const chain = new ConversationChain({
  llm: model,
  prompt,
  memory,
});

// const chain = prompt.pipe(model);

// Goal OF Runnablesequence: The goal of this chain is to process an initial input, load memory (e.g., chat history), and then pass this context to a prompt and model for generating output (//).

const chain1 = RunnableSequence.from([
  {
    input: (initialInput) => initialInput.input,
    memory: () => memory.loadMemoryVariables(),
  },

  {
    input: (previousOutput) => previousOutput.input,
    history: (previousOutput) => previousOutput.memory.history,
  },
  prompt,
  model,
]);

console.log(await memory.loadMemoryVariables());
const response = await chain.invoke({
  input: "Hello",
});
console.log(response);

// update the buffer memory
await memory.saveContext(
  {
    input: "Hello",
  },
  {
    output: response.response,
  }
);

console.log("Updated History", await memory.loadMemoryVariables());

const response1 = await chain.invoke({
  input: "What is Langchain?",
});
console.log(response1);

// update the buffer memory
await memory.saveContext(
  {
    input: "What is Langchain?",
  },
  {
    output: response1.response,
  }
);
