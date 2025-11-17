(async () => {
  // Getting my current page URL to send to the detector API
  const url = window.location.href;

  try {
    // Sending my URL to the backend AI model for prediction
    const res = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url })
    });

    // Getting my JSON response containing the phishing score
    const data = await res.json();

    // Checking my result and showing a banner if the site looks suspicious
    if (data.phishing > 0.5) {
      const banner = document.createElement("div");
      banner.className = "phish-banner";
      banner.textContent = "⚠️  AI detector: likely phishing site";

      // Putting my banner at the top of the page
      document.body.prepend(banner);
    }
  } catch (e) {
    // Handling my network/API failure gracefully
    console.warn("PhishGuard: could not reach API", e);
  }
})();