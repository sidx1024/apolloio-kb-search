from sentence_transformers import SentenceTransformer, util, InputExample, losses
from torch.utils.data import DataLoader
from config import base_model, base_model_dir, model_dir, articles
import json
import os.path

# File paths
articles_file = articles

# Identify cases that aren't being matched correctly
unmatched_cases = [
    ('positive outcomes', "If you use the Apollo engagement suite in your day-to-day workflow, visit the control center to view a synopsis of your most recent engagement stats from the previous, current, and coming week.\nOn the left of your screen, you can see the number of emails delivered in the selected week in comparison to the previous one. You can also see a summary of the percentage of emails your contacts have opened, replied to, unsubscribed from, shown interest in, and that have bounced that week.\nIf you use the dialer to connect with your contacts, you can also see the number of calls connected in the selected week in comparison to the previous one. For more granular insights, you can see the number of calls dialed, the average duration of those calls, the percentage of voicemails you left, and the neutral and positive outcomes you received that week.\nTo compare the more granular email or call stats with the previous week, hover your mouse over the ellipsis icon.\nOn the right, you can visualize your due and completed tasks for the selected week.\nIf you haven't created any tasks yet, click + New task to get started.\n \nMore Info, Please!\nNew to tasks and want a little more context? Hop on down to the \"Tasks\" section below to learn more.\nTo view the statistics of the previous or upcoming week, click the arrow icons.\nIf you don't use the dialer in your engagement strategy, you may want to hide the call data. Click the additional options (...) button and then click Hide call stats .\nYou have now accessed the weekly stats in the control center. Refer to the section below for more information about the To-do list ."),
    ('phone trigger', "In addition to the conditions outlined above, the Call Dispositions you've set up can also be used as Call Trigger stages to set for your Contacts . Refer to the \" Create Call Dispositions to Log Your Calls \" article for more information."),
    ('add contacts to emailer campaign',
     "In Apollo, you can add contacts to a sequence in a few different ways. You can use the Apollo Search tool, upload a CSV file, or select contacts from a list.\nRefer to the sections below for detailed instructions about each method.\n \nOne Trip Per Contact!\nPlease note, each included contact only gets one trip through each sequence step. Apollo tracks progress for contacts to avoid sending duplicate messaging. If you need a contact to go through the same series of sequence steps, clone the original sequence and add the contact to it.\n[Back to Top](#top) ### Add Contacts to a Sequence with the Apollo Search Tool\n \nSo Many Options!\nPlease note, you can add contacts to a sequence using the Apollo search tool from a sequence or directly from a search. The steps remain the same. We just like to give you options.\nLaunch Apollo, click Engage in the navbar, and then click Sequences .\nClick the sequence to which you want to add a contact.\nClick the Add Contacts drop-down.\nClick Prospect Searcher to search for contacts in the Apollo database.\nAdd filters to your search to find the right people for your sequence.\nSelect the contacts that you want to add.\nClick the \"Sequence\" drop-down and then click Add to Sequence .\nNext, click the Sequence drop-down in the Add to Sequence modal and select the sequence to which you would like to add the contact(s).\nClick View all sequences if you want to open a new tab to review all your options from the Sequences page.\nClick Add now to immediately add your contact(s) to the sequence.\nAlternatively, you can schedule to add the list of contacts at a later date. To do so, click Schedule .\nThen, click the radio button next to your preferred scheduled time or manually select a custom date and time for which you want Apollo to add the contacts to the sequence.\nClick Schedule .\n \nVerified Email Credits\nPlease note, verifying data requires a lot of resources. Apollo charges 1 credit per new verified contact; however, there is no charge if you add a saved contact to your sequence for which you already have a verified email address .")
]


# Load or train the model
if os.path.exists(model_dir):
    print('Using existing model from ' + model_dir)
    # Load existing model
    model = SentenceTransformer(model_dir)
else:
    if not os.path.exists(base_model_dir):
        print(base_model_dir + 'does not exist so it will be downloaded and saved')
        model = SentenceTransformer(base_model)
        model.save(model_dir)

    print('Using model from ' + base_model_dir)

    # Define the training procedure
    model = SentenceTransformer(base_model_dir)

    # Load data from JSON file
    with open(articles_file) as f:
        data = (json.load(f))

    # Create InputExamples, with a target similarity of 1.0 for each pair
    examples = []
    for d in data:
        text_a = d['body']
        text_b = ' '.join(d['labels']) + ' ' + \
            ' '.join(d['headings'].split(' > '))
        similarity_score = util.cos_sim(model.encode(
            [text_a]), model.encode([text_b]))[0][0]
        examples.append(InputExample(
            texts=[text_a, text_b], label=similarity_score))

    # Add these cases to your training data
    for case in unmatched_cases:
        text_a = case[0]
        text_b = case[1]
        similarity_score = 0.65  # Set a high similarity score for these cases
        examples.append(InputExample(
            texts=[text_a, text_b], label=similarity_score))

    # Create a DataLoader with your training examples
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)

    train_loss = losses.CosineSimilarityLoss(model)

    # Train the model
    model.fit(train_objectives=[
              (train_dataloader, train_loss)], epochs=2, warmup_steps=100)

    # Save the model
    model.save(model_dir)


# Perform testing using the trained model
test_sentences = ["sales navigator linkedin does not work",
                  "warm up emails", "i opened the email i sent and it shows as opened", "enter a valid email while linking mailbox sid--@gmail.com", "how to resume a contact", "how many email accounts can i add", "phone trigger condition", "positive outcomes", "add contact to sequence", "add contacts to emailer campaign"]

# Encode the test sentences
test_embeddings = model.encode(test_sentences)

# Load the original data
with open(articles_file) as f:
    data = json.load(f)

corpus_embeddings = model.encode([d['body'] for d in data])

# Find the closest match for each test sentence
for i, test_embedding in enumerate(test_embeddings):
    closest_indices = [idx['corpus_id'] for idx in util.semantic_search(
        test_embedding, corpus_embeddings)[0]]
    closest_chunks = [data[idx] for idx in closest_indices][:2]

    print("======================================================================================")
    print("Test Sentence:", test_sentences[i])
    for chunk in closest_chunks:
        similarity_score = util.cos_sim(
            test_embedding, model.encode([chunk['body']]))[0][0]
        print("Similarity Score:", similarity_score)
        print("Title:", chunk['title'])
        print("Body:", chunk['body'])
        print()
