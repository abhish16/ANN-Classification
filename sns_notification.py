import boto3
import os
os.environ["SNSACCESS"] 
os.environ["SNSKEY"] 

# Initialize AWS SNS client
sns = boto3.client('sns', region_name='us-east-1',aws_access_key_id= os.environ.get("SNSACCESS"),aws_secret_access_key=os.environ.get("SNSKEY"))

# Your SNS topic ARN
SNS_TOPIC_ARN = "arn:aws:sns:us-east-1:761018877989:BankChurn"

def send_sns_notification():
    """Publishes a promotional cashback message to SNS."""
    message = "Get 10% cashback up to ₹500 on all Credit Card purchases for 16th and 17th March. Don't miss out!"
    
    response = sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Message=message,
        Subject="Irresistible Cashback"
    )
    print("SNS Notification Sent!", response)