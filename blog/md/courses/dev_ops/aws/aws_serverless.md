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

![Slack app name](../../../../images/dev_ops/aws/Slack-activate-webhook.png)

5. Scroll down and click **Add New Webhook** then choose a channel, and allow

![Slack app name](../../../../images/dev_ops/aws/Slack-add-webhook.png)

6. Store the generated Webhook URL for later
