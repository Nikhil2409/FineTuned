import json
import random

# The number of data points you want to generate.
NUM_EXAMPLES = 1500

# Lists of names to make the emails more realistic
professional_names = [
    "Alex Johnson", "Benjamin Carter", "Catherine Davis", "Daniel Evans", 
    "Emily Foster", "George Miller", "Hannah Wilson", "Jacob Moore",
    "Jessica Taylor", "Kevin Wright", "Laura Thomas", "Michael Clark"
]
casual_names = [
    "Al", "Ben", "Cathy", "Dan", "Em", "George", "Han", "Jake",
    "Jess", "Kev", "Lou", "Mike"
]
your_names = ["Sarah", "John", "Chris"]

def get_random_names():
    """Returns a dictionary of randomly selected names for placeholders."""
    your_name = random.choice(your_names)
    recipient_name = random.choice(professional_names)
    colleague_name = random.choice(professional_names)
    friend_name = random.choice(casual_names)
    manager_name = random.choice(professional_names)
    interviewer_name = random.choice(professional_names)
    client_name = random.choice(professional_names)
    vendor_name = random.choice(professional_names)
    family_member_name = random.choice(casual_names)
    professor_name = random.choice(professional_names)
    
    return {
        "[Your Name]": your_name,
        "[Recipient's Name]": recipient_name,
        "[Colleague's Name]": colleague_name,
        "[Friend's Name]": friend_name,
        "[Manager's Name]": manager_name,
        "[Interviewer's Name]": interviewer_name,
        "[Client's Name]": client_name,
        "[Vendor's Name]": vendor_name,
        "[Family Member's Name]": family_member_name,
        "[Professor's Name]": professor_name,
        "[Sender's Name]": recipient_name  # Use a random professional name for the sender in a formal reply
    }

# Lists of structured data points with a clear instruction-output relationship
email_templates = [
    {
        "instruction": "Draft a professional email to a colleague.",
        "input": "Subject: Project 'Atlas' meeting notes. Body: We discussed the new marketing strategy and next steps.",
        "output": "Hi [Colleague's Name],\n\nI hope you're having a good week. I wanted to follow up on the project discussion we had. Please let me know if you have any questions.\n\nBest,\n[Your Name]"
    },
    {
        "instruction": "Write an informal email to a friend about meeting up.",
        "input": "Subject: Catching up. Body: Haven't seen you in a while! Let's get coffee next week.",
        "output": "Hey [Friend's Name],\n\nIt's been too long! I was wondering if you're free for a coffee sometime next week? Let me know what your schedule looks like.\n\nCheers,\n[Your Name]"
    },
    {
        "instruction": "Draft a follow-up email to a client after a meeting.",
        "input": "Subject: Following up. Body: I'd like to ask a few questions about the interview last week.",
        "output": "Dear [Client's Name],\n\nThank you again for your time last week. I've attached the follow-up documents we discussed and look forward to hearing from you.\n\nSincerely,\n[Your Name]"
    },
    {
        "instruction": "Write a short, polite email requesting information.",
        "input": "Subject: Information Request. Body: I need details on the new product launch.",
        "output": "Dear [Recipient's Name],\n\nI hope this email finds you well. I would like to request some information regarding the new product launch.\n\nThank you for your time and assistance,\n[Your Name]"
    },
    {
        "instruction": "Generate a formal email to a professor.",
        "input": "Subject: Assignment help. Body: I have a question about the homework. Can you help?",
        "output": "Dear Professor [Professor's Name],\n\nI am writing to you today with a question regarding the recent assignment. I would appreciate it if you could provide some guidance. I am available to meet during your office hours.\n\nThank you for your time,\n[Your Name]"
    },
    {
        "instruction": "Create a professional email to an interviewer following up on a job application.",
        "input": "Subject: Following up. Body: I'd like to ask a few questions about the interview last week.",
        "output": "Hello [Interviewer's Name],\n\nThank you for the opportunity to interview for the [Job Title] position. I'm very enthusiastic about the role and am eager to hear about the next steps in the process.\n\nBest regards,\n[Your Name]"
    },
    {
        "instruction": "Draft an email to a team member providing project updates.",
        "input": "Subject: Team update. Body: We're on track to hit our Q3 goals. Good job!",
        "output": "Hi Team,\n\nJust a quick update on the project. We are making great progress and are on schedule. Please continue your hard work.\n\nRegards,\n[Your Name]"
    },
    {
        "instruction": "Write a casual email to a family member.",
        "input": "Subject: Family lunch. Body: Let's meet at Grandma's house this weekend.",
        "output": "Hey Mom,\n\nI'm so excited for lunch at Grandma's this weekend. Can't wait to see you and the family!\n\nLove,\n[Your Name]"
    },
    {
        "instruction": "Draft a professional email to a colleague.",
        "input": "Subject: Project 'Phoenix' Update. Body: I have a new report on the Q4 budget. We need to go over it.",
        "output": "Hi [Colleague's Name],\n\nI hope this email finds you well. I have a new report on the Q4 budget that I would like to discuss with you. Please let me know when you are available to meet.\n\nBest,\n[Your Name]"
    },
    {
        "instruction": "Write a formal email to a new client.",
        "input": "Subject: Sales report. Body: I have attached the sales figures for last month. Please review.",
        "output": "Dear [Client's Name],\n\nI hope this email finds you well. I have attached the sales report for last month for your review. Please let me know if you have any questions.\n\nSincerely,\n[Your Name]"
    },
    {
        "instruction": "Draft a casual email to a friend about a concert.",
        "input": "Subject: Awesome concert last night! Body: You missed an incredible show. The band was amazing!",
        "output": "Hey [Friend's Name],\n\nMan, you really missed out last night! That show was absolutely insane. We have to go to the next one they play!\n\nTalk soon,\n[Your Name]"
    },
    {
        "instruction": "Write a professional email to a manager requesting a day off.",
        "input": "Subject: Leave Request. Body: I would like to request a day off on [Date] for a personal appointment.",
        "output": "Dear [Manager's Name],\n\nI am writing to formally request a day of leave on [Date] due to a personal appointment. I will ensure all my responsibilities are handled before I leave. Thank you for your consideration.\n\nBest regards,\n[Your Name]"
    },
    {
        "instruction": "Draft a friendly email inviting friends to a party.",
        "input": "Subject: Party at my place! Body: I'm hosting a get-together this Saturday. Be there!",
        "output": "Hey everyone,\n\nI'm hosting a party at my place this Saturday at [Time]. I'd love for you to join! There will be food, drinks, and good music.\n\nHope to see you there!\n[Your Name]"
    },
    {
        "instruction": "Write a formal email declining a meeting invitation.",
        "input": "Subject: RE: Meeting on [Date]. Body: I have a prior engagement and will be unable to attend.",
        "output": "Dear [Sender's Name],\n\nThank you for the invitation. Unfortunately, I have a prior commitment and will be unable to attend. I apologize for any inconvenience.\n\nSincerely,\n[Your Name]"
    },
    {
        "instruction": "Draft a professional email to a vendor about a billing issue.",
        "input": "Subject: Billing Inquiry - Invoice #[Number]. Body: We have found an discrepancy on our latest invoice.",
        "output": "Dear [Vendor's Name],\n\nI am writing to you today regarding invoice #[Number]. We have found a discrepancy in the billing amount and would appreciate it if you could review this issue. I have attached the relevant documents for your reference.\n\nThank you,\n[Your Name]"
    },
    {
        "instruction": "Write a casual email to a family member with a recipe.",
        "input": "Subject: Grandma's cookies! Body: I finally got the recipe for Grandma's chocolate chip cookies. Here it is!",
        "output": "Hey [Family Member's Name],\n\nI'm so excited to share Grandma's famous cookie recipe with you. It's a family secret, so don't tell anyone! Let me know how they turn out.\n\nLove,\n[Your Name]"
    },
    {
        "instruction": "Draft a professional email providing a reference for a former employee.",
        "input": "Subject: Reference for [Employee's Name]. Body: [Employee's Name] worked for me for [Number] years as a [Job Title].",
        "output": "To whom it may concern,\n\nI am writing to provide a reference for [Employee's Name]. [He/She] worked under my supervision as a [Job Title] from [Start Date] to [End Date]. I can confirm [he/she] is a diligent and reliable employee.\n\nBest regards,\n[Your Name]"
    },
    {
        "instruction": "Write an informal email to a coworker about lunch plans.",
        "input": "Subject: Lunch? Body: I'm getting hungry. Anyone want to grab a bite at [Time]?",
        "output": "Hey [Coworker's Name],\n\nI'm getting hungry and was wondering if you wanted to grab a bite to eat at [Time]? Let me know if you're free.\n\nCheers,\n[Your Name]"
    },
    {
        "instruction": "Draft a professional email to a client with a project proposal.",
        "input": "Subject: Project Proposal. Body: I've attached a detailed proposal for our new project. We are confident in our ability to deliver on our promises.",
        "output": "Dear [Client's Name],\n\nI hope this email finds you well. I've attached our project proposal for your review. We are confident that this project will meet your needs and exceed your expectations.\n\nSincerely,\n[Your Name]"
    },
    {
        "instruction": "Write a casual email to a friend to ask for a favor.",
        "input": "Subject: A quick favor. Body: I need help moving some furniture this weekend. Can you lend a hand?",
        "output": "Hey [Friend's Name],\n\nI hope you're having a good week. I was wondering if I could ask you for a quick favor? I need a hand moving some furniture this weekend, and I was hoping you could help.\n\nThanks,\n[Your Name]"
    },
    {
        "instruction": "Draft a professional email to a colleague about a new project.",
        "input": "Subject: New project. Body: We have been assigned to a new project. I am looking forward to working with you.",
        "output": "Hi [Colleague's Name],\n\nI hope this email finds you well. I am writing to you today to let you know that we have been assigned to a new project. I am looking forward to working with you on this project.\n\nBest regards,\n[Your Name]"
    },
    {
        "instruction": "Write a casual email to a friend about a movie night.",
        "input": "Subject: Movie night! Body: We are planning a movie night this Friday. What do you think?",
        "output": "Hey [Friend's Name],\n\nI hope you're having a good week. We're planning a movie night this Friday at [Time]. We would love for you to join us! Let me know if you are free.\n\nCheers,\n[Your Name]"
    },
    {
        "instruction": "Draft a professional email to a manager with a progress report.",
        "input": "Subject: Progress report. Body: I have attached the progress report for the Q4 project. We are on track to hit our goals.",
        "output": "Dear [Manager's Name],\n\nI am writing to you today to provide you with the progress report for the Q4 project. I have attached the report for your review. Please let me know if you have any questions.\n\nSincerely,\n[Your Name]"
    },
    {
        "instruction": "Write a formal email to a new client.",
        "input": "Subject: Project discussion. Body: I would like to schedule a call to discuss the project in more detail.",
        "output": "Dear [Client's Name],\n\nI hope this email finds you well. I am writing to you today to schedule a call to discuss the project in more detail. Please let me know what times work best for you.\n\nBest regards,\n[Your Name]"
    },
    {
        "instruction": "Draft a professional email to a colleague about a new client.",
        "input": "Subject: New client. Body: We have a new client. I would like to introduce you to them.",
        "output": "Hi [Colleague's Name],\n\nI hope this email finds you well. I am writing to you today to let you know that we have a new client, [Client's Name]. I would like to introduce you to them.\n\nBest regards,\n[Your Name]"
    },
    {
        "instruction": "Write a casual email to a friend about a coffee date.",
        "input": "Subject: Coffee date. Body: I would love to catch up with you. Let's get coffee next week.",
        "output": "Hey [Friend's Name],\n\nIt's been too long! I would love to catch up with you and hear about what you've been up to. Are you free to grab coffee sometime next week?\n\nCheers,\n[Your Name]"
    },
    {
        "instruction": "Draft a professional email to a client with a follow-up question.",
        "input": "Subject: Follow-up question. Body: I have a quick question about the project. Can you help me?",
        "output": "Dear [Client's Name],\n\nI hope this email finds you well. I am writing to you today with a quick follow-up question regarding the project. Please let me know if you have a moment to discuss.\n\nBest regards,\n[Your Name]"
    },
    {
        "instruction": "Write a formal email to a supplier about a new order.",
        "input": "Subject: New order. Body: I would like to place a new order for the [Product Name]. Please send me the details.",
        "output": "Dear [Supplier's Name],\n\nI am writing to you today to place a new order for the [Product Name]. Please send me the details and a quote for the order. I look forward to hearing from you soon.\n\nSincerely,\n[Your Name]"
    },
    {
        "instruction": "Draft a professional email to a manager with a suggestion.",
        "input": "Subject: Suggestion for improvement. Body: I have a suggestion for improving our workflow.",
        "output": "Dear [Manager's Name],\n\nI hope this email finds you well. I would like to share a suggestion for improving our workflow. I have attached a document with more details for your review. Thank you for your consideration.\n\nSincerely,\n[Your Name]"
    },
    {
        "instruction": "Write a casual email to a family member about a new movie.",
        "input": "Subject: New movie! Body: I just saw a new movie and it was great. We should watch it together.",
        "output": "Hey [Family Member's Name],\n\nI hope you're having a good week. I just saw a new movie and it was great! We should all watch it together sometime soon.\n\nLove,\n[Your Name]"
    }
]

# Generate the dataset by repeating the templates
dataset = []
# Create a number of examples equal to NUM_EXAMPLES by randomly selecting templates
for _ in range(NUM_EXAMPLES):
    template = random.choice(email_templates)
    
    # Create a new entry and fill in the placeholders with random names
    new_entry = template.copy()
    
    names = get_random_names()
    for placeholder, name in names.items():
        # Replace placeholders in the output
        new_entry["output"] = new_entry["output"].replace(placeholder, name)

    dataset.append(new_entry)

# Shuffle the final dataset to ensure a random order
random.shuffle(dataset)

# Save the dataset to a JSON file
with open('email_data.json', 'w') as f:
    json.dump(dataset, f, indent=4)

print(f"✅ Successfully generated and saved {len(dataset)} email examples to 'email_data.json'.")
