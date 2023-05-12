import { JSDOM } from 'jsdom';

const dom = new JSDOM('<html></html>');
const { Node } = dom.window;

export async function processData(rawArticles) {
  const articles = rawArticles.filter((article) => !article.draft);
  const chunks = [];
  for (const article of articles) {
    const articleData = getArticleData(article);
    for (const articleChunk of articleData) {
      chunks.push(articleChunk);
    }
  }

  return chunks;
}

function getArticleData(article) {
  const document = JSDOM.fragment(article.body);

  return getHeadingChunks(document).map((chunk) => {
    return {
      ...chunk,
      title: article.title,
      labels: article.label_names,
      html_url: chunk.heading_id
        ? article.html_url + '#' + chunk.heading_id
        : article.html_url,
      created_at: article.created_at,
    };
  });
}

function getArticleDataOld(article) {
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
        headings = headings.join(' > ');
        if (data.at(-1)?.headings === headings) {
          data.at(-1).body += text + ' ';
        } else {
          data.push({
            // type: node.nodeName,
            title: article.title,
            headings,
            labels: article.label_names,
            body: text + ' ',
            html_url: article_url,
            created_at: article.created_at,
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

function getHeadingChunks(node) {
  let chunks = [];
  let chunk = { headings: [], heading_id: '', text: '' };
  let headingHierarchy = [];

  function processNode(node) {
    if (node.nodeType === Node.DOCUMENT_FRAGMENT_NODE) {
      for (let child of node.childNodes) {
        processNode(child);
      }
    } else if (node.nodeType === Node.TEXT_NODE) {
      chunk.text += ' ' + node.nodeValue;
    } else if (node.nodeType === Node.ELEMENT_NODE) {
      if (/^h[1-6]$/i.test(node.tagName)) {
        let currentHeadingLevel = parseInt(node.tagName.substring(1));
        headingHierarchy = headingHierarchy.slice(0, currentHeadingLevel - 1);
        headingHierarchy.push(naiveInnerText(node).trim());
        if (chunk.headings.length !== 0 || chunk.text.trim() !== '') {
          chunks.push({ ...chunk });
          chunk = { headings: [...headingHierarchy], text: '' };
        } else {
          chunk.headings = [...headingHierarchy];
        }
        chunk.heading_id = node.id;
      } else if (node.tagName.toLowerCase() === 'a') {
        if (node.getAttribute('href') !== '#top') {
          chunk.text += ' ' + naiveInnerText(node);
        }
      } else if (node.tagName.toLowerCase() === 'li') {
        chunk.text += ' - ';
        for (let child of node.childNodes) {
          processNode(child);
        }
      } else {
        for (let child of node.childNodes) {
          processNode(child);
        }
      }
    }
  }

  processNode(node);

  if (chunk.headings.length !== 0 || chunk.text.trim() !== '') {
    chunks.push(chunk);
  }

  return chunks.map((chunk) => {
    return {
      headings: chunk.headings.join(' > '),
      heading_id: chunk.heading_id,
      body: chunk.text
        .replaceAll(/\t/g, '\n')
        .replaceAll(/[ ]*[\n][ ]*/g, '\n')
        .replaceAll(/\n{1,}/g, '\n')
        .replaceAll(/ {2,}/g, ' ')
        .trim(),
    };
  });
}

function naiveInnerText(node) {
  const Node = node; // We need Node(DOM's Node) for the constants, but Node doesn't exist in the nodejs global space, and any Node instance references the constants through the prototype chain
  return [...node.childNodes]
    .map((node) => {
      switch (node.nodeType) {
        case Node.TEXT_NODE:
          return node.textContent;
        case Node.ELEMENT_NODE:
          return naiveInnerText(node);
        default:
          return '';
      }
    })
    .join('\n');
}
