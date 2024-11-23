Office.onReady(() => {
  // If needed, Office.js is ready to be called
});

function checkPhishing(event) {
  Office.context.mailbox.getCallbackTokenAsync(
    { isRest: true },
    async function (result) {
      if (result.status === "succeeded") {
        const accessToken = result.value;
        const itemId = getItemRestId(); // Get the item's REST ID.
        // Construct the REST URL to the current item. /$value returns the email data in EML
        const eml =
          Office.context.mailbox.restUrl +
          "/v2.0/me/messages/" +
          itemId +
          "/$value";
        const evaluateEmailURL =
          "https://trisapple.pythonanywhere.com/evaluateEmail";

        // Get Email Data in EML
        const emlRequest = await fetch(eml, {
          headers: {
            Authorization: "Bearer " + accessToken,
          },
        });
        const emlResponse = await emlRequest.text(); // Email in EML

        // Send EML to our Python Anywhere Server in the request body. Determine if email is 'safe' or 'not safe'
        const parseEML = await fetch(evaluateEmailURL, {
          method: "POST",
          body: emlResponse,
        });
        // Our server will return a response -> 'The email is safe' or 'The email is not safe'
        const parseEMLResponse = await parseEML.text();

        const message = {
          type: Office.MailboxEnums.ItemNotificationMessageType
            .InformationalMessage,
          message: parseEMLResponse,
          icon: "Icon.80x80",
          persistent: true,
        };

        // Show a notification message
        Office.context.mailbox.item.notificationMessages.replaceAsync(
          "action",
          message
        );
        event.completed(); // Indicate that the add-in command function is complete
      } else {
        // Handle the error.
        const message = {
          type: Office.MailboxEnums.ItemNotificationMessageType
            .InformationalMessage,
          message: "Unable to get callback token.",
          icon: "Icon.80x80",
          persistent: true,
        };

        // Show a notification message
        Office.context.mailbox.item.notificationMessages.replaceAsync(
          "action",
          message
        );
        event.completed(); // Indicate that the add-in command function is complete
      }
    }
  );
}

function getItemRestId() {
  if (Office.context.mailbox.diagnostics.hostName === "OutlookIOS") {
    // itemId is already REST-formatted.
    return Office.context.mailbox.item.itemId;
  } else {
    // Convert to an item ID for API v2.0.
    return Office.context.mailbox.convertToRestId(
      Office.context.mailbox.item.itemId,
      Office.MailboxEnums.RestVersion.v2_0
    );
  }
}

function getGlobal() {
  return typeof self !== "undefined"
    ? self
    : typeof window !== "undefined"
    ? window
    : typeof global !== "undefined"
    ? global
    : undefined;
}

const g = getGlobal();

// The add-in command functions need to be available in global scope
g.action = checkPhishing;
