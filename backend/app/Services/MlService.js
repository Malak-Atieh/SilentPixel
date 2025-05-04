const axios = require('axios');

async function callPythonML({ action, image, message, password, watermark, generateQR }) {
  const formData = new FormData();
  formData.append('image', image, 'image.png');
  if (message) formData.append('message', message);
  if (password) formData.append('password', password);
  if (watermark) formData.append('watermark', watermark);
  if (generateQr) formData.append('generateQr', generateQr);

  const endpoint = action === 'encode' ? '/encode' : '/decode';

  const response = await axios.post(`http://localhost:5000/api/${endpoint}`, formData, {
    headers: formData.getHeaders()
  });

  return response.data;
}

module.exports = { callPythonML };
