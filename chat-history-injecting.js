import * as dotenv from "dotenv";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { Document } from "@langchain/core/documents";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { FakeEmbeddings } from "langchain/embeddings/fake";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";

dotenv.config();

async function createVectorStore() {
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

  //   // Instantiate the Embeddings
  //   const embedding = new GoogleGenerativeAIEmbeddings({
  //     apiKey: process.env.GEMINI_API_KEY,
  //     modelName: "embedding-001",
  //   });

  const embedding = new FakeEmbeddings();

  // Instantiate the Vector store to save into the memory
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splittedDocs,
    embedding
  );

  return vectorStore;
}

async function createChain() {
  // Instantaite the model
  const model = new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    temperature: 0.7,
    apiKey: process.env.GEMINI_API_KEY,
  });

  //   // Create the prompt using template
  //   const prompt = ChatPromptTemplate.fromTemplate(`
  //     Answer the user question.
  //     Question: {input},
  //     ChatHistory: {chat_history},
  //     Context: {context}
  //     `);

  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Answer the user's question based on following context: {context}",
    ],
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
  ]);

  // Chaining the prompt with the model
  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt,
  });

  // Now retrieve the Data from the vector store
  const retriever = vectorStore.asRetriever({
    k: 2,
  });

  const retriavalPrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
      "user",
      "GIven the above conversation, generate a search query to look up in order get information relevent to the conversation",
    ],
  ]);

  const historyAwareRetriever = await createHistoryAwareRetriever({
    llm: model,
    retriever,
    rephrasePrompt: retriavalPrompt,
  });

  const retrivalChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever: historyAwareRetriever,
  });

  return retrivalChain;
}

const vectorStore = await createVectorStore();
const retrievalChainNew = await createChain(vectorStore);

const chatHistory = [
  new HumanMessage("Hello"),
  new AIMessage("How may I help you?"),
  new HumanMessage("I am Musta"),
  new AIMessage("Hello Musta, how may I help you?"),
  new HumanMessage("What is Multimodality?"),
  new AIMessage(
    "Multimodality refers to the presentation of information in multiple formats or modes. Instead of relying on just one way to communicate (like text), multimodality uses a combination of different elements"
  ),
];

// Invoking the chain
const res = await retrievalChainNew.invoke({
  input: "what is multimodality",
  chat_history: chatHistory,
});
console.log(res);

/*
What is the difference between "ChatPromptTemplate.fromMessages" and "ChatPromptTemplate.fromTemplate":
Answer:  Both are used to create prompts for chat models (like Gemini, GPT-4, etc.), but they are designed for different use cases.
         fromTemplate --> It treats the whole prompt as a single message. 
         from Messages --> structure the prompt as multiple chat messages, structured chat messages with roles. 

What is the functionality of historyAwareRetriever:
Answer:  Normal retrieval doesn't feed the chatHistory. It only retrieves the data from vector that relevent to query. To feed the input as well as 
         chatHistory, historyAwareRetriever is required.

New queries and answer from the AI can be pushed to the chatHistory List. 

*/
