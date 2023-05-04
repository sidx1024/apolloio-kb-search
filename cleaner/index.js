const fs = require('fs');
const { convertArrayToCSV } = require('convert-array-to-csv');
const parse5 = require('parse5');
const { JSDOM } = require('jsdom');

const CHUNK_SIZE = 512; //
const inputFilename = './raw_articles.json';
const outputFilename = './clean_articles.json';
// const outputFilename = './clean_articles.csv';

const input = JSON.parse(fs.readFileSync(inputFilename, 'utf-8'));

const chunks = [];
for (const article of input) {
  const articleData = getArticleData(article);
  for (const articleChunk of articleData) {
    chunks.push(...splitChunk(articleChunk));
  }
}

function splitChunk(chunk) {
  let chunks = [];
  const parts = Math.ceil(chunk.body.length / CHUNK_SIZE);

  let currentIndex = 0;

  for (let i = 0; i < parts; i++) {
    let endIndex = currentIndex + CHUNK_SIZE;

    if (endIndex < chunk.body.length) {
      while (chunk.body[endIndex] !== ' ' && endIndex > currentIndex) {
        endIndex--;
      }
    } else {
      endIndex = chunk.body.length;
    }

    chunks.push({
      ...chunk,
      body: chunk.body.slice(currentIndex, endIndex),
    });

    currentIndex = endIndex + 1;
  }

  return chunks;
}

console.log(chunks);

// const content = convertArrayToCSV(chunks);
content = chunks;
fs.writeFileSync(outputFilename, JSON.stringify(content, null, 2), 'utf-8');

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
        const headings = headingStack.map((h) => h.textContent).join(' > ');
        if (data.at(-1)?.headings === headings) {
          data.at(-1).body += text + ' ';
        } else {
          data.push({
            // type: node.nodeName,
            headings,
            body: text + ' ',
            html_url: article_url,
          });
        }
      }
    }
    // console.log(
    //   `${' '.repeat(depth * 2)}Node type: ${node.nodeName}, ${
    //     node.nodeName === '#text' ? node.textContent : ''
    //   }`
    // );
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

// function getInnerText(htmlString) {
//   if (typeof htmlString !== 'string') {
//     return '';
//   }
//   const documentFragment = parse5.parse(htmlString);
//   const bodyNode = findNodeByName(documentFragment, 'body');
//   const innerText = extractInnerText(bodyNode);
//   return innerText.trim();
// }

// // Custom function to traverse the DOM tree and build the innerText
// function extractInnerText(node) {
//   if (node.nodeName === '#text') {
//     return node.value.trim() + ' ';
//   }

//   let innerText = '';
//   if (node.childNodes) {
//     node.childNodes.forEach((childNode) => {
//       innerText += extractInnerText(childNode);
//     });
//   }

//   return innerText;
// }

// // Custom function to find a node by name in the DOM tree
// function findNodeByName(node, name) {
//   if (node.nodeName === name) {
//     return node;
//   }

//   if (node.childNodes) {
//     for (const childNode of node.childNodes) {
//       const foundNode = findNodeByName(childNode, name);
//       if (foundNode) {
//         return foundNode;
//       }
//     }
//   }

//   return null;
// }
