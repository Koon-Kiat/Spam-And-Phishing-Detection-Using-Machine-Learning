/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT license.
 * See LICENSE in the project root for license information.
 */

/* global global, Office, self, window */

Office.onReady(() => {
  // If needed, Office.js is ready to be called
});

/**
 * Shows a notification when the add-in command is executed.
 * @param event {Office.AddinCommands.Event}
 */
function action(event) {
  const message = {
    type: Office.MailboxEnums.ItemNotificationMessageType.InformationalMessage,
    message: "Performed action.",
    icon: "Icon.80x80",
    persistent: true,
  };

  // Show a notification message
  Office.context.mailbox.item.notificationMessages.replaceAsync("action", message);

  // Be sure to indicate when the add-in command function is complete
  event.completed();
}

function checkPhishing(event) {

  Office.context.mailbox.getCallbackTokenAsync({isRest: true}, async function(result){
    if (result.status === "succeeded") {
      const accessToken = result.value;
  
      // Use the access token.
      // subject = getCurrentItem(accessToken)

      // Get the item's REST ID.
      const itemId = getItemRestId();

      // Construct the REST URL to the current item.
      // Details for formatting the URL can be found at
      // https://learn.microsoft.com/previous-versions/office/office-365-api/api/version-2.0/mail-rest-operations#get-messages.
      const getMessageUrl = Office.context.mailbox.restUrl + '/v2.0/me/messages/' + itemId; // Email URL
      const eml = Office.context.mailbox.restUrl + '/v2.0/me/messages/' + itemId + '/$value'; // EML file
      const evaluateEmailURL = "https://trisapple.pythonanywhere.com/evaluateEmail"
      console.log(getMessageUrl)
      console.log(accessToken)

      const emlRequest = await fetch(eml, {
        headers: {
          'Authorization': 'Bearer ' + accessToken
        }
      })
      const emlResponse = await emlRequest.text()

      const parseEML = await fetch(evaluateEmailURL, {
        method: "POST",
        body: emlResponse
      })
      const parseEMLResponse = await parseEML.text()
      console.log(parseEMLResponse)

      const response = await fetch(getMessageUrl, {
        headers: {
          'Authorization': 'Bearer ' + accessToken
        }
      })
      const json = await response.json()
      const subject = json.Subject;
      const message = {
        type: Office.MailboxEnums.ItemNotificationMessageType.InformationalMessage,
        message: parseEMLResponse,
        icon: "Icon.80x80",
        persistent: true,
      };
  
      // Show a notification message
      Office.context.mailbox.item.notificationMessages.replaceAsync("action", message);
  
      // Be sure to indicate when the add-in command function is complete
      event.completed();
      
    } else {
      // Handle the error.
    }
  });
}

function getItemRestId() {
  if (Office.context.mailbox.diagnostics.hostName === 'OutlookIOS') {
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
g.action = action, checkPhishing;
