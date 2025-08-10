// Grab DOM elements (declare once lang)
const chatbox = document.getElementById('chatbox');
const inputBox = document.getElementById('inputBox');
const sendBtn = document.getElementById('sendBtn');

// Training data
const trainingData = [
  // Greetings
  { input: 'hello', output: 'greeting' },
  { input: 'hi', output: 'greeting' },
  { input: 'hey', output: 'greeting' },
  { input: 'how are you', output: 'greeting' },
  { input: "what's up", output: 'greeting' },
  { input: 'good morning', output: 'greeting' },
  { input: 'good afternoon', output: 'greeting' },
  { input: 'good evening', output: 'greeting' },
  { input: 'hiya', output: 'greeting' },
  { input: 'yo', output: 'greeting' },

  // Farewells
  { input: 'bye', output: 'farewell' },
  { input: 'goodbye', output: 'farewell' },
  { input: 'see you', output: 'farewell' },
  { input: 'talk to you later', output: 'farewell' },
  { input: 'see ya', output: 'farewell' },
  { input: 'catch you later', output: 'farewell' },
  { input: 'take care', output: 'farewell' },

  // Thanks
  { input: 'thanks', output: 'thanks' },
  { input: 'thank you', output: 'thanks' },
  { input: 'thank you very much', output: 'thanks' },
  { input: 'thank you so much', output: 'thanks' },
  { input: 'thanks a lot', output: 'thanks' },
  { input: 'much appreciated', output: 'thanks' },

  // Capabilities
  { input: 'what can you do?', output: 'capabilities' },
  { input: 'what are your capabilities?', output: 'capabilities' },
  { input: 'what can you help me with?', output: 'capabilities' },
  { input: 'tell me what you can do', output: 'capabilities' },
  { input: 'help me', output: 'help' },
  { input: 'can you help me?', output: 'help' },
  { input: 'i need assistance', output: 'help' },
  { input: 'i need help', output: 'help' },

  // Add some common questions or chit-chat
  { input: 'how is the weather?', output: 'weather' },
  { input: 'what time is it?', output: 'time' },
  { input: 'tell me a joke', output: 'joke' },
];

// Responses
const responses = {
  greeting: ['Hello! How can I help you?', 'Hi there!', "Hey! What's up?"],
  farewell: ['Goodbye!', 'See you later!', 'Bye! Take care!'],
  thanks: ['You’re welcome!', 'No problem!', 'Anytime! Happy to help!'],
  capabilities: [
    'I am a simple AI chatbot built with TensorFlow.js. I can greet, say goodbye, and thank you!',
    'I can understand simple greetings, farewells, and thanks.',
    'I can also try to help with basic questions.',
  ],
  help: ['Sure! Ask me anything within my capabilities.', "I'm here to assist you. What do you want to know?"],
  weather: ['I cannot check the weather yet, but maybe check a weather website?', 'Sorry, I do not have live weather data yet.'],
  time: ['I do not have access to the current time, sorry!', 'Check your device clock for the time.'],
  joke: ['Why did the computer show up at work late? It had a hard drive!', 'Why do programmers prefer dark mode? Because light attracts bugs!'],
  unknown: ['Sorry, I did not understand that. Can you try rephrasing?', "I'm not sure I follow. Could you say that differently?"],
};

// Tokenizer with simple stemming
function tokenize(text) {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^\w\s]/gi, '')
    .split(/\s+/)
    .map(word => {
      if (word.endsWith('es')) return word.slice(0, -2);
      if (word.endsWith('s')) return word.slice(0, -1);
      return word;
    });
}

// Build vocabulary once
const vocabulary = [];
trainingData.forEach(item => {
  tokenize(item.input).forEach(word => {
    if (!vocabulary.includes(word)) vocabulary.push(word);
  });
});

// Vectorize text
function vectorizeText(text) {
  const tokens = tokenize(text);
  return vocabulary.map(word => tokens.includes(word) ? 1 : 0);
}

// Prepare training data tensors
const xs = tf.tensor2d(trainingData.map(item => vectorizeText(item.input)));
const labelsList = Array.from(new Set(trainingData.map(d => d.output)));
const labels = trainingData.map(item => labelsList.indexOf(item.output));
const ys = tf.oneHot(labels, labelsList.length);

// Build model
const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [vocabulary.length], units: 16, activation: 'relu' }));
model.add(tf.layers.dense({ units: 12, activation: 'relu' }));
model.add(tf.layers.dense({ units: labelsList.length, activation: 'softmax' }));

model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

// Train model
async function trainModel() {
  await model.fit(xs, ys, { epochs: 150 });
  console.log('Model trained');
}
trainModel();

// Context memory
let lastIntent = null;

// Chat function
async function chat(input) {
  const inputVectorArr = vectorizeText(input);
  console.log('Tokens:', tokenize(input));
  console.log('Vocabulary:', vocabulary);
  console.log('Input vector:', inputVectorArr);

  const inputVector = tf.tensor2d([inputVectorArr]);
  const prediction = model.predict(inputVector);

  const predictedIndex = (await prediction.argMax(-1).data())[0];
  const confidence = (await prediction.max().data())[0];

  console.log('Predicted index:', predictedIndex, 'Confidence:', confidence);

  inputVector.dispose();
  prediction.dispose();

  let intent = labelsList[predictedIndex];
  if (confidence < 0.3) {
    intent = 'unknown';
  }

  lastIntent = intent;

  const possibleResponses = responses[intent] || responses['unknown'];
  const botReply = possibleResponses[Math.floor(Math.random() * possibleResponses.length)];
  return botReply;
}

// Add message to chatbox
function addMessage(sender, text) {
  const div = document.createElement('div');
  div.classList.add('chat-message', sender);
  div.textContent = text;
  chatbox.appendChild(div);
  chatbox.scrollTop = chatbox.scrollHeight;
}

// Event listeners
sendBtn.addEventListener('click', async () => {
  const userText = inputBox.value.trim();
  if (!userText) return;

  addMessage('user', userText);
  inputBox.value = '';

  const botReply = await chat(userText);
  addMessage('bot', botReply);
});

inputBox.addEventListener('keydown', e => {
  if (e.key === 'Enter') sendBtn.click();
});
