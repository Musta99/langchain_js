import * as dotenv from "dotenv";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
dotenv.config();

const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash",
  temperature: 1,
  apiKey: process.env.GEMINI_API_KEY,
});

const prompt = ChatPromptTemplate.fromTemplate(
  "You are a joker. Tell a joke based on the following word {input}"
);

const chain = prompt.pipe(model);

const res = await chain.invoke({ input: "Dog" });

console.log(res.content);
