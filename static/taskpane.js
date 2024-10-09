/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT license.
 * See LICENSE in the project root for license information.
 */

/* global document, Office */

Office.onReady((info) => {
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

  if (info.host === Office.HostType.Outlook) {
    document.getElementById("sideload-msg").style.display = "none";
    document.getElementById("app-body").style.display = "flex";
    document.getElementById("output").innerHTML = 'Getting email data...';

    Office.context.mailbox.getCallbackTokenAsync({isRest: true}, async function(result) {
      if (result.status === "succeeded") {
        const accessToken = result.value;
        const itemId = getItemRestId(); // Get the item's REST ID.
        // Construct the REST URL to the current item. /$value returns the email data in EML
        const eml = Office.context.mailbox.restUrl + '/v2.0/me/messages/' + itemId + '/$value';
        const evaluateEmailURL = "https://trisapple.pythonanywhere.com/evaluateEmail"
  
        // Get Email Data in EML
        const emlRequest = await fetch(eml, {
          headers: {
            'Authorization': 'Bearer ' + accessToken
          }
        })
        const emlResponse = await emlRequest.text() // Email in EML
  
        document.getElementById("output").innerHTML = 'Determining if email is safe or phishing...';
        // Send EML to our Python Anywhere Server in the request body. Determine if email is 'safe' or 'not safe'
        const parseEML = await fetch(evaluateEmailURL, {
          method: "POST",
          body: emlResponse
        })
        // Our server will return a response -> 'The email is safe' or 'The email is not safe'
        const parseEMLResponse = await parseEML.text()
  
        document.getElementById("output").innerHTML = parseEMLResponse;
      } else {
        // Handle the error.
        document.getElementById("output").innerHTML = "Unable to get email data."
      }
    });
  }
});

/**
export async function run() {
  
   * Insert your Outlook code here
   
}
*/
