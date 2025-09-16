import * as dotenv from "dotenv";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Document } from "@langchain/core/documents";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { FakeEmbeddings } from "langchain/embeddings/fake";
dotenv.config();

/*

What we want is, we want our model to answer the questions from any 
database, any documents or any website by providing the URL of that 
specific website.

Langchain Documents: A documents is an object that contains text, methods
                     PDF Name, URL etc.

In order to pass the documents to chain, we need to develop a special 
types of chain called CreateStuffDocumentysChain.

As we can't generate so many documents for our website manually, 
instead we shouldfetch data from a website and then those data from 
website will automatically be converted to documents by document 
loader called Cheerio tool. 
------------- > run "npm install cheerio"  < -----------------

When we provide a URL, loader extracts a lot of informations from the website 
and we want to have only the relevent information we require to show. For that we have
to break the info into small chunks which we do by using textSplitter. And then to get the 
most relevent answer, we have to store those documents into vector store.

The diagram is given below: 
Source --> Data(lots of data that contain relevent and irrelevent contents) --> Breaks the data into small
chunks --> Embed the data in order to store into vector store --> Retrieve from the store.

Embedding: Embedding is a type language that the computer or AI understands



*/

// Instantaite the model
const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash",
  temperature: 0.7,
  apiKey: process.env.GEMINI_API_KEY,
});

// Create the prompt template
const prompt = ChatPromptTemplate.fromTemplate(`
    Answer the user question. 
    Question: {input},
    Context: {context}
    `);

// // Create a single document Manually
// const documentA = new Document({
//   pageContent: "Today is 16th September, 2025 and It is 11:14 AM Now",
// });

// const documentB = new Document({
//   pageContent: "Tomorrow is 17th September, 2025 and It is 11:14 AM Now",
// });

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

// Instantiate the Embeddings
const embedding = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  modelName: "embedding-001",
});

// const embedding = new FakeEmbeddings();

// Instantiate the Vector store to save into the memory
const vectorStore = await MemoryVectorStore.fromDocuments(
  splittedDocs,
  embedding
);

// Chaining the prompt with the model
const chain = await createStuffDocumentsChain({
  llm: model,
  prompt,
});

// Now retrieve the Data from the vector store
const retriever = vectorStore.asRetriever({
  k: 2,
});

const retrivalChain = await createRetrievalChain({
  combineDocsChain: chain,
  retriever,
});

// Invoking the chain
const res = await retrivalChain.invoke({
  input: "What is Multimodality",
});

console.log(res.answer);
