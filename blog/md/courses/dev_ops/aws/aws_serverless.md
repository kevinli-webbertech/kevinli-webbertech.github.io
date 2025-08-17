##  AWS Serverless Project – GitHub Actions, Lambda & Slack Integration

### Prerequisites

- An AWS account

![Aws Account homepage](../../../../images/dev_ops/aws/Aws-account.png)

- A github repository for testing
  
![Empty Github repo](../../../../images/dev_ops/aws/Github-repo.png)

- Access to a Slack account and workspace where you can create an *Incoming Webhook*.

![Slack Workspace example](../../../../images/dev_ops/aws/Slack-workspace.png)

## Creating an AWS Account

1. Go to [https://aws.amazon.com/](https://aws.amazon.com/) and **Create an AWS Account**.
2. Fill in **contact information** and select account type (Personal is fine).
3. Add a **credit/debit card** for verification (Free Tier is fine).
4. Verify your identity via SMS or automated call
5. Select the **Basic** support plan (free).
6. Log in at [https://console.aws.amazon.com](https://console.aws.amazon.com).

## Setting up a Slack incoming Webhook

1. Go to [Slack Incoming Webhooks](https://api.slack.com/messaging/webhooks).
2. Click **Create your Slack app** → **From scratch**.

![Create Slack app btn](../../../../images/dev_ops/aws/Create-slack-app.png)

3. Give it a name and select your workspace

![Slack app name](../../../../images/dev_ops/aws/Slack-app-name.png)

4. In **Incoming Webhooks**, toggle ON to activate.

![Slack activate Webhook](../../../../images/dev_ops/aws/Slack-activate-webhook.png)

5. Scroll down and click **Add New Webhook** then choose a channel, and allow

![Slack add a new webhook](../../../../images/dev_ops/aws/Slack-add-webhook.png)

1. Store the generated Webhook URL for later

## 1 - Create AWS Lambda Function

1. In AWS Console, go to Lambda -> Create Function

![Lambda Create Function](../../../../images/dev_ops/aws/Lambda-create-function.png)

2. Select:
   - Author from scratch
   - Give it a Name: `slack-notifier`
   - Runtime: Python 3.13
   - Architecture: x86_64
  
![Lambda Function Settings](../../../../images/dev_ops/aws/Lambda-function-settings.png)

3. Click on **Create function**

4. Add environment variables

   - Choose the Configuration tab, then choose Environment variables.
   - Under Environment variables, choose Edit.
   - Add environment variables and dsave them: 
     - `SLACK_WEBHOOK_URL` = your Slack webhook URL `SHARED_TOKEN` = a simple secret (secret string inveted by you) to validate GitHub requests 
   
![Lambda Environment variables](../../../../images/dev_ops/aws/Lambda-environment-variables.png)



