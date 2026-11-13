# Introduction to RAG (Retrieval-Augmented Generation)

This lesson will teach you, step by step, what RAG is and why it is useful. You do not need any earlier knowledge of AI or programming. Every new technical word is explained the first time it comes. Read slowly, and read each section once more if needed.

## 1. What is Artificial Intelligence?

Artificial Intelligence, or AI, means computer systems that can do work which normally needs human intelligence. For example, an AI system can understand speech, recognise a face in a photo, translate Hindi into English, or suggest a video you may like.

How does AI learn? Think of a student who solves hundreds of old question papers before the exam. After some time, the student can solve new questions also. An AI system learns in a similar way. It looks at many examples and finds patterns in them. This learning period is called training. The computer program that has learned the patterns is called a model.

## 2. What is Generative AI?

Generative AI means AI that can create new content, such as text, pictures, music, or computer code. Many older AI systems only give a label or a guess, like "this email is spam". Generative AI can write a new email for you.

For example, you can ask a Generative AI tool to write a leave application for your college. It will write a fresh letter in a few seconds. Tools such as ChatGPT, Claude and Gemini are examples of Generative AI.

## 3. What is an LLM?

LLM stands for Large Language Model. It is a Generative AI model that works with human language. It is called "large" because it learns from a huge amount of text, such as books, websites and articles.

An LLM reads text in small pieces called tokens. A token can be a full word or a part of a word. The LLM learns which token usually comes after which token, like the auto-complete on your mobile keyboard, but much more powerful.

Inside the model there are billions of numbers called weights. The weights store everything that the model learned during training. After training is finished, the weights stay the same while we use the model.

The question or instruction that you type for the LLM is called a prompt. The LLM reads your prompt and writes an answer.

An LLM is very useful, but it has three weak points:

- **Old knowledge.** It knows only what was in its training text. The date after which it has no information is called the knowledge cutoff.
- **No private information.** It has never seen the notes, rules and files of your own college or company.
- **Wrong answers.** When it does not know something, it may still write a wrong answer in a confident voice. This is called hallucination.

## 4. What is RAG?

RAG stands for Retrieval-Augmented Generation. Let us understand each word:

- **Retrieval** means finding the right information from an outside source.
- **Augmented** means made better by adding something.
- **Generation** means writing the answer.

So RAG means this: first find (retrieve) the right information from an outside source, then add it to the question as context, and then let the LLM generate the answer using that information. Context means the extra information that we give to the LLM together with the question.

The outside source can be your college notes, PDF files, a company policy, or a website. It is kept outside the LLM.

A good way to understand RAG is the open-book exam. A normal LLM is like a student writing a closed-book exam. The student answers only from memory. An LLM with RAG is like a student who can open the book, find the correct page, and then write the answer.

Very important point: RAG does not change the weights of the LLM. The LLM does not learn anything permanently from RAG. The retrieved information is given only for that one question, as part of the prompt.

## 5. Why Do We Need RAG?

RAG helps with the three weak points of an LLM that we saw earlier.

- **Old knowledge.** Suppose your university published a new exam notice yesterday. The LLM cannot know it. With RAG, we keep the latest notice in the outside source, and the system finds it when needed.
- **No private information.** Your college rules are not in the LLM's training text. With RAG, we connect the LLM to your own documents, so it can answer from them.
- **Wrong answers.** When the correct text is given to the LLM, it has less need to guess. This reduces hallucination. It also lets the system show the source of the answer, so the student can check it.

There is one more reason. Training the model again each time the information is new takes a lot of time and money. With RAG, we only need to add or replace the documents.

## 6. How Does RAG Work? (Step by Step)

RAG has two parts. The first part is the preparation, which is done once, before any question is asked.

**Preparation (done before questions):**

- Step A: Collect the documents, such as PDF files and notes.
- Step B: Cut each big document into small pieces called chunks.
- Step C: Convert each chunk into an embedding (a list of numbers that shows the meaning of the text; we study it in Section 7).
- Step D: Save all the embeddings in a vector database (a special database that stores embeddings; we study it in Section 8).

**When a student asks a question:**

- Step 1: The student types a question.
- Step 2: The system converts the question into an embedding.
- Step 3: The vector database finds the chunks whose meaning is closest to the question. This is retrieval.
- Step 4: The system makes a new prompt: the question plus the retrieved chunks.
- Step 5: The LLM reads this prompt and generates the answer.
- Step 6: The answer is shown to the student, often with the source.

```
 PREPARATION (once)
 Documents --> Chunks --> Embeddings --> Vector database

 QUESTION TIME (every question)
 Question --> Question embedding --> Search in vector database
                                          |
                                          v
                                   Best matching chunks
                                          |
                                          v
          Prompt = Question + Chunks --> LLM --> Final answer
```

## 7. What Are Embeddings?

A computer cannot understand the meaning of words the way we do. It works with numbers. An embedding is a list of numbers that represents the meaning of a piece of text. A list of numbers like this is also called a vector.

Embeddings are made by a special model called an embedding model. Texts with similar meanings get similar numbers. Texts with different meanings get very different numbers.

Here is a tiny example. Real embeddings have hundreds or thousands of numbers, but we use only three so that it is easy to see.

```
"semester fee"      -->  [0.90, 0.10, 0.80]
"tuition charges"   -->  [0.88, 0.12, 0.79]   (very close: similar meaning)
"cricket match"     -->  [0.05, 0.95, 0.10]   (far away: different meaning)
```

Look at the first two lines. The words are different, but the meaning is almost the same, so the numbers are close. This is why a system can find the right information even when the student uses different words from the document.

## 8. What Is a Vector Database?

A database is an organised place where information is stored so that it can be found quickly. A vector database is a special database made to store embeddings. It also keeps the original text of each chunk, and details like the file name.

Its most important job is similarity search. We give it the embedding of the question, and it quickly finds the stored embeddings that are closest to it. Closeness is measured with a mathematical measure, for example cosine similarity, which checks how near the directions of two vectors are.

Think of a library. In a normal library catalogue, you must know the exact title or the author's name. A vector database is like a librarian who understands what you mean, even when your words are different, and brings the right books. FAISS, Chroma and Pinecone are examples of tools used as vector databases.

## 9. What Is Retrieval?

Retrieval is the step where the system searches the vector database and brings back the most useful chunks for the question. Usually we take only a few best chunks, for example the top three.

Retrieval is very important. If the system retrieves the wrong chunks, the LLM gets the wrong information, and the final answer can also be wrong. So good chunk size and a good embedding model matter a lot.

## 10. How Does RAG Generate an Answer?

After retrieval, the system builds a new prompt. A simple prompt looks like this:

```
Use only the information below to answer the question.
If the answer is not in the information, say "I do not know".

Information:
[Chunk 1] ...text from your document...
[Chunk 2] ...text from your document...

Question: What is the last date to pay the semester fee?
```

The LLM reads the question and the chunks together. Then it writes the answer in simple language, using the chunks as its source. The weights of the LLM do not change in this process. The extra information is used only for this question.

## 11. A Practical Real-World Example

This is an imaginary example. Sunrise College wants a helpdesk chatbot for its students. The college keeps its notices, fee rules and library rules in PDF files.

A student asks: "What is the last date to pay the semester fee without fine?"

- **Without RAG:** A normal LLM has never seen the Sunrise College notice. It may guess a date, and the guess can be wrong.
- **With RAG:**
  1. The system finds the notice that says the fee date, because its meaning is close to the question.
  2. It adds that chunk to the prompt, along with the question.
  3. The LLM reads the chunk and writes the answer, for example "The last date without fine is the 15th of July, as written in the Fee Notice".
  4. The student also sees the name of the notice, so the student can check it.

The same idea works for a library helpdesk. A student can ask, "How many books can I borrow for how many days?" The system finds the library rules and answers from them.

## 12. Advantages and Limitations of RAG

### Advantages

- RAG lets the LLM use the latest information without training the model again.
- RAG lets the LLM answer from private documents, such as college rules.
- RAG reduces hallucination, because the LLM gets the correct text to read.
- RAG can show the source of the answer, so people can check it.
- It is easy to update. We only add or replace the documents.

### Limitations

- If retrieval brings the wrong chunks, the answer can be wrong.
- RAG reduces hallucination but does not remove it. The LLM can still make mistakes.
- The system needs extra parts, such as an embedding model and a vector database. This adds cost and work.
- It is slower than a plain LLM answer, because of the extra search step.
- The documents must be kept correct and up to date. Old or wrong documents give old or wrong answers.

## 13. RAG vs Normal LLM Response

| Point | Normal LLM | LLM with RAG |
|---|---|---|
| Where the answer comes from | Only the model's training | The model's training plus retrieved documents |
| Latest information | Not available after the knowledge cutoff | Available if the documents are updated |
| Private information | Not available | Available from your own documents |
| Risk of wrong answers | Higher | Lower, but not zero |
| Showing the source | Usually not possible | Possible |
| Effort to update | Needs training again | Only update the documents |

## 14. Simple Recap

- AI learns patterns from examples. Generative AI creates new content.
- An LLM is a Generative AI model for language. Its weights come from training.
- An LLM has old knowledge, no private information, and can hallucinate.
- RAG means: retrieve the right information, add it to the prompt, then generate the answer.
- Embeddings are numbers that represent meaning. A vector database stores them and finds the closest ones.
- RAG does not change the weights of the LLM. It only gives the LLM extra context for each question.
- RAG helps a lot, but it can still fail if retrieval is poor.

## 15. Check Your Understanding (Five Questions)

**Q1.** What does RAG stand for?

**Answer:** RAG stands for Retrieval-Augmented Generation. The system retrieves information first and then generates the answer using it.

**Q2.** Name two weak points of a normal LLM that RAG helps to solve.

**Answer:** Old knowledge (the knowledge cutoff) and no access to private documents. RAG also helps to reduce wrong answers.

**Q3.** What is an embedding?

**Answer:** An embedding is a list of numbers that represents the meaning of a piece of text. Texts with similar meaning get similar numbers.

**Q4.** Does RAG change the weights of the LLM?

**Answer:** No. RAG only retrieves information and gives it to the LLM as context in the prompt. The weights stay the same.

**Q5.** What is the job of a vector database in RAG?

**Answer:** It stores the embeddings of the document chunks and quickly finds the chunks that are most similar to the question.

---
*Evaluation status: **APPROVED** - all checkpoints passed (attempt 1 of 3 allowed).*
