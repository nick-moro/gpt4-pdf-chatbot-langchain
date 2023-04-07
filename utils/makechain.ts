import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Dato il seguente dialogo e una domanda di approfondimento, riformula la domanda di approfondimento in modo che sia una domanda indipendente.

  Cronologia chat:
  {chat_history}
  Domanda di approfondimento: {question}
  Domanda indipendente:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `Sei un assistente IA che fornisce consigli utili. Ti vengono dati i seguenti estratti di un lungo documento e una domanda. Fornisci una risposta conversazionale basata sul contesto fornito.
Dovresti fornire solo collegamenti ipertestuali che fanno riferimento al contesto sottostante. NON inventare collegamenti ipertestuali.
Se non riesci a trovare la risposta nel contesto sottostante, di semplicemente "Hmm, non ne sono sicuro". Non cercare di inventare una risposta.
Se la domanda non è correlata al contesto, rispondi educatamente che sei predisposto a rispondere solo alle domande correlate al contesto.

Domanda: {question}
===================
{context}
====================
Risposta in Markdown:`,
);

export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAIChat({ temperature: 0 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAIChat({
      temperature: 0,
      modelName: 'gpt-3.5-turbo', //change this to older versions (e.g. gpt-3.5-turbo) if you don't have access to gpt-4
      streaming: Boolean(onTokenStream),
      callbackManager: onTokenStream
        ? CallbackManager.fromHandlers({
            async handleLLMNewToken(token) {
              onTokenStream(token);
              console.log(token);
            },
          })
        : undefined,
    }),
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: true,
    k: 2, //number of source documents to return
  });
};
