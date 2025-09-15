import * as dotenv from "dotenv";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {
  StringOutputParser,
  CommaSeparatedListOutputParser,
} from "@langchain/core/output_parsers";

import { StructuredOutputParser } from "langchain/output_parsers";

import { z } from "zod";
dotenv.config();

const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash",
  temperature: 1,
  apiKey: process.env.GEMINI_API_KEY,
});

// --------------- String output Parse -----------------

const promptString = ChatPromptTemplate.fromTemplate(
  "You are a joker. Tell a joke based on the following word {input}"
);

const parserString = new StringOutputParser();

const chainString = promptString.pipe(model).pipe(parserString);

const resString = await chainString.invoke({
  input: "rainy day",
});

// console.log(resString);

//  ---------------- Comma separated List Output Parser --------------------------

const promptList = ChatPromptTemplate.fromTemplate(`
    Provide me 5 antonyms, separated by comma, of the following word {word}
    `);

const parserList = new CommaSeparatedListOutputParser();

const chainList = promptList.pipe(model).pipe(parserList);

const resList = await chainList.invoke({
  word: "sleep",
});

// console.log(resList);

// ----------------- Structured Output Parser ----------------
const promptStructured = ChatPromptTemplate.fromTemplate(`
    Extract the name, age, and class from the following phrase.
    phrase: {phrase}
    formating instruction: {format}
    `);

const structuredParser = StructuredOutputParser.fromNamesAndDescriptions({
  name: "name of the person",
  age: "age of the person",
  class: "in which class, the person studies",
});

const chain = promptStructured.pipe(model).pipe(structuredParser);

const structuredRes = await chain.invoke({
  phrase: "This is mustafiz of 30 years old and he is in class 8",
  format: structuredParser.getFormatInstructions(),
});

// console.log(structuredRes);

// ------------ Structured Output Parser using Zod Schema --------------------
const promptStructuredZod = ChatPromptTemplate.fromTemplate(`
    Extract the recipe name, ingredients list and chef name from the following phrase.
    phrase: {phrase}
    formating instruction: {format}
    `);

const zodFormat = StructuredOutputParser.fromZodSchema(
  z.object({
    name: z.string().describe("Recipe name"),
    list: z.array(z.string()).describe("Ingredients List"),
    chefName: z.string().describe("Chef name"),
  })
);

const chainZod = promptStructuredZod.pipe(model).pipe(zodFormat);

const zodRes = await chainZod.invoke({
  phrase:
    "Mr. Abdullah is cooking fried rice using rice, egg, vegetables, oil and fresh chicken",
  format: zodFormat.getFormatInstructions(),
});

console.log(zodRes);
