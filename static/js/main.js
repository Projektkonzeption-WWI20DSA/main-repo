document.getElementById('text-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form submission
    
    const formData = new FormData(this);
    const inputText = formData.get('input-text'); // Get the text from the input box
  
    // Make an HTTP POST request to the server for summary
    fetch('/process-summary', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text: inputText })
    })
    .then(response => response.json())
    .then(data => {
      // Handle the summary response from the server, if needed
      console.log(data);
      document.getElementById('summary-text').innerText = data.summary;
    })
    .catch(error => {
      // Handle any errors that occur during the summary request
      console.error('Error:', error);
    });
  
    // Make an HTTP POST request to the server for classification
    fetch('/process-classification', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text: inputText })
    })
    .then(response => response.json())
    .then(data => {
      // Handle the classification response from the server, if needed
      console.log(data);
      document.getElementById('classification-text').innerText = data.classification;
    })
    .catch(error => {
      // Handle any errors that occur during the classification request
      console.error('Error:', error);
    });
  });
  