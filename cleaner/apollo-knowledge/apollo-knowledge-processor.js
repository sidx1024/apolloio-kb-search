import { JSDOM } from 'jsdom';
import { splitChunk } from '../utils.js';

export async function processData(articles) {
  const chunks = [];
  for (const article of articles) {
    const articleData = getArticleData(article);
    for (const articleChunk of articleData) {
      chunks.push(...splitChunk(articleChunk));
    }
  }

  return chunks;
}

function getArticleData(article) {
  const html = article.body;
  const data = [];
  const headingStack = [];

  // Parses the HTML and creates a document object.
  const document = JSDOM.fragment(html);

  // Recursive function to traverse through all nodes.
  function traverseNodes(node, depth = 0) {
    // Print node information.
    if (['H1', 'H2', 'H3', 'H4', 'H5', 'H6'].includes(node.nodeName)) {
      const level = Number(/\d/.exec(node.nodeName));
      const currentHeading = headingStack.at(-1);
      if (currentHeading) {
        const currentHeadingLevel = Number(/\d/.exec(currentHeading.nodeName));
        if (currentHeadingLevel >= level) {
          headingStack.pop();
        }
      }
      headingStack.push(node);
    }

    if (
      node.nodeName !== '#text' &&
      node.nodeName !== '#document-fragment' &&
      node.nodeName !== 'SCRIPT' &&
      node.nodeName !== 'BODY' &&
      node.nodeName !== 'HTML' &&
      node.nodeName !== 'LI' &&
      !node.querySelector('h1, h2, h3, h4, h5, h6')?.length
    ) {
      const text = node.textContent?.trim();
      if (typeof text === 'string' && text.length && text !== 'Back to Top') {
        const headingBookmark = headingStack.at(-1)?.id;
        let article_url = article.html_url;
        if (headingBookmark) {
          article_url += '#' + headingBookmark;
        }
        let headings = headingStack.map((h) => h.textContent);
        if (headings[0] !== article.title) {
          headings.unshift(article.title);
        }
        if (data.at(-1)?.headings === headings) {
          data.at(-1).body += text + ' ';
        } else {
          data.push({
            // type: node.nodeName,
            title: article.title,
            headings,
            body: text + ' ',
            html_url: article_url,
          });
        }
      }
    }
    // Traverse child nodes if they exist.
    if (node.childNodes && node.childNodes.length > 0) {
      for (const child of Array.from(node.childNodes)) {
        traverseNodes(child, depth + 1);
      }
    }
  }

  // Call the traverseNodes function with the root document object.
  traverseNodes(document);

  return data;
}
