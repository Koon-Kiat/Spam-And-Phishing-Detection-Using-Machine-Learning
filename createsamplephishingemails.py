import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def create_sample_phishing_email(subject, sender, recipient, body, file_path):
    """
    Creates a sample phishing email in .eml format.
    
    Args:
        subject (str): Subject of the email.
        sender (str): The sender's email address.
        recipient (str): The recipient's email address.
        body (str): The body content of the phishing email.
        file_path (str): The file path to save the .eml file.
    """
    # Create email message
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = recipient

    # Attach body to the email
    body_part = MIMEText(body, 'plain')
    msg.attach(body_part)

    # Save email to .eml file
    with open(file_path, 'w') as file:
        file.write(msg.as_string())

    print(f"Phishing email saved to: {file_path}")

# Directory to save phishing email samples
email_directory = "phishing_emails"
os.makedirs(email_directory, exist_ok=True)

# Create multiple phishing email samples
emails = [
    {
        'subject': "Verify Your Account Immediately!",
        'sender': "no-reply@bank-secure.com",
        'recipient': "user123@domain.com",
        'body': "Dear user,\n\nWe've detected suspicious activity in your account. Please verify your account by clicking the link below to avoid suspension.\n\n[https://www.neck.sample.org/]"
    },
    {
        'subject': "Payment Confirmation Needed!",
        'sender': "billing@online-payments.com",
        'recipient': "customer@domain.com",
        'body': "Dear customer,\n\nYour recent payment was not processed. Please confirm your billing details by clicking the following link to complete your transaction.\n\n[http://sample.org/bankpayment]"
    },
    {
        'subject': "Security Alert: Unusual Login Attempt",
        'sender': "security@trusted-domain.com",
        'recipient': "user@domain.com",
        'body': "Dear user,\n\nWe have detected a login attempt from an unfamiliar location. To secure your account, please click the link below and update your login details.\n\n[http://www.sample.org/login-update]"
    },
    {
        'subject': "Claim Your Free Gift Now!",
        'sender': "promo@rewardscenter.com",
        'recipient': "victim@domain.com",
        'body': "Congratulations!\n\nYou have been selected to receive an exclusive reward! Click the link below to claim your free gift.\n\n[https://sample.edu/day]"
    },
    {
        'subject': "Your Account Has Been Compromised!",
        'sender': "alert@secureaccess.com",
        'recipient': "user@domain.com",
        'body': "Dear user,\n\nYour account has been compromised. Please reset your password immediately by following the link below.\n\n[https://sample.net/#reset-password]"
    },
    {
        'subject': "URGENT: Confirm Your Email Address Now!",
        'sender': "support@fakeservice.com",
        'recipient': "someone@domain.com",
        'body': "Dear valued user,\n\nWe noticed unusual activity on your account. Please confirm your email by clicking the link below to prevent deactivation.\n\n[https://sample.org/confirm-email]"
    },
    {
        'subject': "Delivery Failed: Update Your Address",
        'sender': "shipping@deliveryservice.com",
        'recipient': "user@domain.com",
        'body': "Dear customer,\n\nYour package could not be delivered due to an incorrect address. Please update your delivery details by following the link below.\n\n[https://sample.edu/delivery-update]"
    },
    {
        'subject': "Congratulations! You Won a Free iPhone!",
        'sender': "prize@promotions.com",
        'recipient': "winner@domain.com",
        'body': "Dear user,\n\nYou have won a brand new iPhone! Click the link below to claim your prize.\n\n[http://sample.org/#winprize]"
    },
    {
        'subject': "Your Subscription is About to Expire",
        'sender': "subscriptions@streamingservice.com",
        'recipient': "user@domain.com",
        'body': "Dear user,\n\nYour subscription to our service is about to expire. Renew your subscription now by clicking the link below to continue enjoying our services.\n\n[http://sample.info/?insect=fireman&porter=attraction#cave]"
    },
    {
        'subject': "Action Required: Account Verification",
        'sender': "verification@fakeservice.com",
        'recipient': "user@domain.com",
        'body': "Dear user,\n\nYour account requires verification. Please click the link below to complete the verification process.\n\n[http://www.example.com/?badge=boot]"
    },
    {
        'subject': "Important: Your Invoice is Overdue",
        'sender': "invoices@fakebusiness.com",
        'recipient': "customer@domain.com",
        'body': "Dear customer,\n\nYour invoice is overdue. Please make payment by clicking the link below to avoid late fees.\n\n[http://example.com/balance.html]"
    }
]

# Generate phishing email samples
for i, email in enumerate(emails):
    file_path = os.path.join(email_directory, f"phishing_email_{i+1}.eml")
    create_sample_phishing_email(email['subject'], email['sender'], email['recipient'], email['body'], file_path)
