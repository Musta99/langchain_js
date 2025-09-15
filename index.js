// import { Document } from "@langchain/core/documents";
// import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

// const text = `Some other considerations include:

// - Do you deploy your backend and frontend together, or separately?
// - Do you deploy your backend co-located with your database, or separately?

// **Production Support:** As you move your LangChains into production, we'd love to offer more hands-on support.
// Fill out [this form](https://airtable.com/appwQzlErAS2qiP0L/shrGtGaVBVAz7NcV2) to share more about what you're building, and our team will get in touch.

// ## Deployment Options

// See below for a list of deployment options for your LangChain app. If you don't see your preferred option, please get in touch and we can add it to this list.`;

// const splitter = new RecursiveCharacterTextSplitter({
//   chunkSize: 50,
//   chunkOverlap: 1,
//   separators: ["|", "##", ">", "-"],
// });

// const output = await splitter.splitDocuments([
//   new Document({ pageContent: text }),
// ]);

// console.log(output.slice(0, 3));

// import { GoogleGenerativeAI } from "@google/generative-ai";

// const genAI = new GoogleGenerativeAI("AIzaSyCjvXsHBPdTx7QehwLxu_2tG4X4VlpXEG0");
// const model = genAI.getGenerativeModel({ model: "gemini-2.5-pro", t });

// const prompt = "How much updated you are?";
// const result = await model.generateContent(prompt);
// const response = await result.response;
// const text = response.text();

// console.log("✅ Gemini Response:", text);



import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash",
  temperature: 1,
  apiKey: "AIzaSyCjvXsHBPdTx7QehwLxu_2tG4X4VlpXEG0",
});

const prompt = ChatPromptTemplate.fromTemplate(
  "You are a joker. Tell a joke based on the following word {input}"
);

const chain = prompt.pipe(model);

const res = await chain.invoke({ input: "Dog" });

console.log(res.content);


