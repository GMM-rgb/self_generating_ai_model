const express = require('express');
const axios = require('axios');
const path = require('path');
const { styleText } = require('util');
const chalk = require('chalk'); // This will work with chalk@4.1.2

const app = express();
const PORT = 3000;

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.post('/chat', async (req, res) => {
    const { message, history } = req.body;

    // Create the prompt by combining history and the new message
    const prompt = history.map(entry => entry.user).join('\n') + '\n' + message;

    try {
        const response = await axios.post('http://localhost:11434/api/generate', {
            model: 'llama2',
            prompt,
            options: {
                temperature: 0.4,
                top_p: 1.0
            }
        }, { responseType: 'stream' });

        let completeResponse = '';

        response.data.on('data', (chunk) => {
            const decodedChunk = chunk.toString('utf-8');
            const jsonLines = decodedChunk.trim().split('\n');

            jsonLines.forEach((jsonLine) => {
                if (jsonLine.trim() === '') return;

                let parsedLine;
                try {
                    parsedLine = JSON.parse(jsonLine);
                } catch (error) {
                    console.error('JSON parse error:', error);
                    return;
                }

                if (parsedLine.response) {
                    completeResponse += parsedLine.response;
                }

                if (parsedLine.done) {
                    res.json({ response: completeResponse.trim() });
                    console.log(chalk.green(`${completeResponse}`));
                }
            });
        });

        response.data.on('end', () => {
            if (!completeResponse) {
                res.json({ response: 'No response from the AI model.' });
            }
        });

    } catch (error) {
        res.json({ response: `Error connecting to AI model: ${error.message}` });
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
