import React from "react";
import "./styles.css";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github.css";
import remarkSubSuper from "remark-sub-super";
// import hljs from "highlight.js";
export default function App() {
  const input =
    'OpenAI\'s responses can be formatted in various ways depending on the context and the nature of the query. Here are some common formats with examples:\n\n 1. **Plain Text:**\n   - **Example:**\n     ```\n     The capital of France is Paris.\n     ```\n\n2. **Bullet Points:**\n   - **Example:**\n     ```\n     Here are some benefits of exercise:\n     - Improves mental health\n     - Boosts physical fitness\n     - Enhances mood\n     ```\n\n3. **Numbered Lists:**\n   - **Example:**\n     ```\n     Steps to bake a cake:\n     1. Preheat the oven to 350°F (175°C).\n     2. Mix the dry ingredients.\n     3. Add the wet ingredients.\n     4. Pour the batter into a pan.\n     5. Bake for 30 minutes.\n     ```\n\n4. **Code Blocks:**\n   - **Example:**\n     ```python\n     def greet(name):\n         return f"Hellooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo, {name}!"\n\n     print(greet("Alice"))\n     ```\n\n5. **Tables:**\n   - **Example:**\n     ```\n     | Country | Capital    | Population (millions) |\n     |---------|------------|-----------------------|\n     | France  | Paris      | 67                    |\n     | Germany | Berlin     | 83                    |\n     | Italy   | Rome       | 60                    |\n     ```\n\n6. **Quotes:**\n   - **Example:**\n     ```\n     Albert Einstein once said, "Imagination is more important than knowledge."\n     ```\n\n7. **Headings and Subheadings:**\n   - **Example:**\n     ```\n     ## Benefits of Meditation\n\n     ### Mental Health\n     - Reduces stress\n     - Improves focus\n\n     ### Physical Health\n     - Lowers blood pressure\n     - Enhances immune function\n     ```\n\n8. **Dialogues:**\n   - **Example:**\n     ```\n     Person A: How are you today?\n     Person B: I\'m doing well, thank you! How about you?\n     ```\n\n9. **Q&A Format:**\n   - **Example:**\n     ```\n     Q: What is the tallest mountain in the world?\n     A: Mount Everest.\n     ```\n\n10. **JSON:**\n    - **Example:**\n      ```json\n      {\n          "name": "Alice",\n          "age": 30,\n          "city": "New York"\n      }\n      ```\n\nThese formats help in presenting information clearly and effectively based on the user\'s needs and the type of information being conveyed.';
  return (
    <>
      <div className="App">
        <h1>Hello CodeSandbox</h1>
        <h2>Start editing to see some magic happen!</h2>
        {/* <ReactMarkdown>{input}</ReactMarkdown> */}
      </div>
      <div className="App">
        <h1>Hello CodeSandbox</h1>
        <h2>Start editing to see some magic happen!</h2>

        {/* Markdown rendering with code highlighting */}
        <ReactMarkdown rehypePlugins={[rehypeHighlight]}>{input}</ReactMarkdown>

        {/* Custom <pre> tag that also uses Highlight.js */}
        <pre>
          <code className="language-javascript">
            {`// Hello World example
console.log("Hello, World!");`}
          </code>
        </pre>
      </div>
    </>
  );
}
