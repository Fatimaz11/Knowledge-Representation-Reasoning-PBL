function predict() {
    const temp = document.getElementById("temp").value;
    const humid = document.getElementById("humid").value;

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            temperature: temp,
            humidity: humid
        })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerHTML =
            `<h3>Comfort Score: ${data.comfort_score}</h3>
             <h3>Level: ${data.comfort_level}</h3>`;
    })
    .catch(error => {
        document.getElementById("result").innerText = "Error connecting to server";
    });
}
