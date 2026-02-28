// Paste this into DevTools Console on claude.ai (F12 → Console)
// Then copy the output and save it as cookies.json in the mistral folder

(() => {
  const cookies = {};
  document.cookie.split("; ").forEach(c => {
    const [name, ...rest] = c.split("=");
    cookies[name] = rest.join("=");
  });
  const json = JSON.stringify(cookies, null, 2);
  copy(json);  // copies to clipboard
  console.log("Cookies copied to clipboard! Paste into cookies.json");
  console.log(json);
})();
