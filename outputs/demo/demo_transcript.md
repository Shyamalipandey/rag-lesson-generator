# Deliberate Error Demonstration

- Run ID: `20261113-143824-d0156b`
- LLM provider: `offline`
- Final status: **APPROVED** after 2 attempt(s)

## 1. Initial (incorrect) lesson
Demonstration mode inserted this wrong sentence on purpose into attempt 1:

> RAG means that an LLM permanently learns new information by changing its model weights.

Context inside the lesson:

```
- **No private information.** It has never seen the notes, rules and files of your own college or company.
- **Wrong answers.** When it does not know something, it may still write a wrong answer in a confident voice. This is called hallucination.

## 4. What is RAG?

RAG means that an LLM permanently learns new information by changing its model weights.

RAG stands for Retrieval-Augmented Generation. Let us understand each word:

- **Retrieval** means finding the right information from an outside source.
- **Augmented** means made better by adding something.
- **Generation** means writing the answer.
```

## 2. Evaluator result for attempt 1
Overall status: **FAIL**

- FAIL - Technical accuracy
- PASS - Beginner-friendly language
- PASS - Clear explanation of RAG
- PASS - Explanation of why RAG is needed
- PASS - Explanation of the complete RAG workflow
- PASS - Correct explanation of embeddings
- PASS - Correct explanation of vector databases
- PASS - Practical real-world examples
- PASS - No unexplained technical jargon
- PASS - Logical teaching sequence
- PASS - Advantages and limitations
- PASS - Knowledge assessment questions
- PASS - Required sections, RAG vs normal LLM, and recap

## 3. Failed checkpoint(s) and rejection reason
- **Technical accuracy**: Incorrect claim: "RAG means that an LLM permanently learns new information by changing its model weights.". RAG retrieves relevant external information and supplies it as context to the LLM. It does not normally update the model's weights.

## 4. Regeneration
The regeneration agent received the rejected lesson, the failed checkpoints and the suggestions, and wrote attempt 2.
The wrong sentence is gone in the new lesson.

## 5. Corrected lesson (the part about weights)
```
The outside source can be your college notes, PDF files, a company policy, or a website. It is kept outside the LLM.

A good way to understand RAG is the open-book exam. A normal LLM is like a student writing a closed-book exam. The student answers only from memory. An LLM with RAG is like a student who can open the book, find the correct page, and then write the answer.

Very important point: RAG does not change the weights of the LLM. The LLM does not learn anything permanently from RAG. The retrieved information is given only for that one question, as part of the prompt.

## 5. Why Do We Need RAG?

RAG helps with the three weak points of an LLM that we saw earlier.

- **Old knowledge.** Suppose your university published a new exam notice yesterday. The LLM cannot know it. With RAG, we keep the latest notice in the outside source, and the system finds it when needed.
```

## 6. Evaluation of attempt 2
Overall status: **PASS**

- PASS - Technical accuracy
- PASS - Beginner-friendly language
- PASS - Clear explanation of RAG
- PASS - Explanation of why RAG is needed
- PASS - Explanation of the complete RAG workflow
- PASS - Correct explanation of embeddings
- PASS - Correct explanation of vector databases
- PASS - Practical real-world examples
- PASS - No unexplained technical jargon
- PASS - Logical teaching sequence
- PASS - Advantages and limitations
- PASS - Knowledge assessment questions
- PASS - Required sections, RAG vs normal LLM, and recap

## Final result
**APPROVED**
