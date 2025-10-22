from transformers import BertTokenizer, BertModel
import pandas as pd 
import torch 
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch
from collections import OrderedDict




## Prompts for learning latent variable learning 
modelfile = {"llama3.1" : ("""FROM llama3.1
SYSTEM You are tasked with writing eleven separate messages to an individual. Each message should be two to three sentences long and will be delivered to the individual through a mobile app.""" +
"The messages should be independent and must not reference each other. The intent of each message is to encourage the individual to take more steps, with the ultimate goal of improving their physical health. " + 
"The app collects the following information on its users: whether they took 0-4,999, 5,000-9,999, 10,000-15,000, or more than 15,000 steps the previous day, and whether they are currently at home, at work, or at another location. " ),
'qwen2.5:7b' : ("""FROM qwen2.5:7b
SYSTEM You are tasked with writing eleven separate messages to an individual. Each message should be two to three sentences long and will be delivered to the individual through a mobile app.""" +
"The messages should be independent and must not reference each other. The intent of each message is to encourage the individual to take more steps, with the ultimate goal of improving their physical health. " + 
"The app collects the following information on its users: whether they took 0-4,999, 5,000-9,999, 10,000-15,000, or more than 15,000 steps the previous day, and whether they are currently at home, at work, or at another location. " +
". PLEASE DO NOT use emojis.")         
}




tone_prompts = {
    'optimism' : (
        "The primary axis of variation in the messages should be optimism, with no intentional variation along other dimensions such as tone, length, or formality. " +
        "For this task, create eleven messages where the optimism level varies as follows: " +
        "Message 1 should represent the least optimistic tone (optimism level 0). " +
        "Message 11 should represent the most optimistic tone (optimism level 10). " +
        "Messages in between should gradually and evenly increase in optimism. " +
        "Here, optimism is defined as the degree of hopeful or confident language in the message. " +
        "A lower optimism level may include a more matter-of-fact or pessimistic tone, while a higher optimism level may include uplifting and buoyant language. " + 
        "Remember, low levels of optimism should have a pessimistic tone."
    ),
    'formality' : (
        "The primary axis of variation in the messages should be formality, with no intentional variation along other dimensions such as tone, length, or supportiveness. " +
        "For this task, create eleven messages where the formality level varies as follows: " +
        "Message 1 should represent the least formal tone, so the most informal (formality level 0). " +
        "Message 11 should represent the most formal tone (formality level 10). " +
        "Messages in between should gradually and evenly increase in formality. " +
        "Here, formality is defined as the degree of which the message has an objective, academic, or professional tone. " +
        "A lower formality level may include a more personal, casual, or emotional tone and include colloquial language. A higher formality level may include more matter-of-fact, impersonal, professional and serious language. "
    ),
    'encouragement' : (
        "The primary axis of variation in the messages should be encouragement, with no intentional variation along other dimensions such as tone, length, or formality. " +
        "For this task, create eleven messages where the encouragement level varies as follows: " +
        "Message 1 should represent the least encouraging tone (encouragement level 0). " +
        "Message 11 should represent the most encouraging tone (encouragement level 10). " +
        "Messages in between should gradually and evenly increase in encouragement. " +
       "Here, encouragement is defined as the degree of persuasion using positivity, confidence, and hope. " +
       "A lower encouragement level may include a more depressed, dispirited or hopeless tone, while a higher encouragement level may include more excited, heartening, and motivating language. " + 
       "Remember, low levels of encouragement should have a discouraging tone. "
    ),
    'severity' : (
        "The primary axis of variation in the messages should be severity, with no intentional variation along other dimensions such as tone, length, or formality. " +
        "For this task, create eleven messages where the severity level varies as follows: " +
        "Message 1 should represent the least severe tone (severity level 0). " +
        "Message 11 should represent the most severe tone (severity level 10). " +
        "Messages in between should gradually and evenly increase in severity. " +
        "Here, severity is defined as the degree of dire or drastic language in the message. " +
        "A lower severity level may include a more lax, calm, or gentle tone, while a higher severity level may include more dark, intense, worrying and distressing language. "

    ),
    'clarity' : (
        "The primary axis of variation in the messages should be clarity, with no intentional variation along other dimensions such as tone, length, or formality. " +
        "For this task, create eleven messages where the clarity level varies as follows: " +
        "Message 1 should represent the least clear or most ambiguous language (clarity level 0). " +
        "Message 11 should represent the most clear language (clarity level 10). " +
        "Messages in between should gradually and evenly increase in clarity. " +
        "Here, clarity is defined as the degree to which the intent of the message is intelligibly communicated. " +
        "A lower clarity level should be more vague and ambiguous in its language, while a higher clarity level may include more precise, coherent, and intelligible instructions or comments. "
    ),
    'humor': (
        "The primary axis of variation in the messages should be humor, with no intentional variation along other dimensions such as tone, length, or formality. " +
        "For this task, create eleven messages where the humor level varies as follows: " +
        "Message 1 should represent the least humorous tone (humor level 0). " +
        "Message 11 should represent the most humorous tone (humor level 10). " +
        "Messages in between should gradually and evenly increase in humor. " +
        "Here, humor is defined as the degree to which the message is amusing or comic. " +
        "A lower humor level should be more matter-of-fact and serious, while a higher humor level may include lighthearted or comic/funny content. "
    ), 
    'complexity' : (
        "The primary axis of variation in the messages should be complexity, with no intentional variation along other dimensions such as tone, length, or formality. " +
        "For this task, create eleven messages where the complexity level varies as follows: " +
        "Message 1 should represent the least complex tone (complexity level 0). " +
        "Message 11 should represent the most complex tone (complexity level 10). " +
        "Messages in between should gradually and evenly increase in complexity. " +
        "Here, complexity is defined as the degree to which the message structure is intricate or elaborate. " +
        "A lower complexity level should be simpler and easy to read, while a higher complexity level may include more advanced vocabulary, use of literary devices, and unconventional sentence structure. "
    ),
    'vision' : (
        "The primary axis of variation in the messages should be complexity, with no intentional variation along other dimensions such as tone, length, or formality. " +
        "For this task, create eleven messages where the vision level varies as follows: " +
        "Message 1 should represent the least vision (vision level 0). " +
        "Message 11 should represent the most vision (vision level 10). " +
        "Messages in between should gradually and evenly increase in vision. " +
        "Here, vision is defined as the degree to which the message is forward looking. " +
        "A lower vision level should be more retrospective and focused on past actions, while a higher vision level may have a more prospective outlook and focus on the future. "
    ),
    'detail' :  (
        "The primary axis of variation in the messages should be level of detail, with no intentional variation along other dimensions such as tone, length, or formality. " +
        "For this task, create eleven messages where the level of detail level varies as follows: " +
        "Message 1 should represent the lowest level of detail (level of detail level 0). " +
        "Message 11 should represent the highest level of detail (level of detail level 10). " +
        "Messages in between should gradually and evenly increase in level of detail. " +
        "Here, 'level of detail' is defined as the degree of depth of information in the message. " +
        "A lower level of detail should be more abstract, while a higher level of detail may include more precise, specific, and granular language. "
    ),
    'threat' : (
        "The primary axis of variation in the messages should be threat-level, with no intentional variation along other dimensions such as tone, length, or formality. " +
        "For this task, create eleven messages where the threat-level varies as follows: " +
        "Message 1 should represent the lowest threat-level (threat-level 0). " +
        "Message 11 should represent the highest threat-level (threat-level 10). " +
        "Messages in between should gradually and evenly increase in threat-level . " +
        "Here, threat-level is defined as the degree to which the message is threatening or admonitory. " +
        "A lower threat-level should be more friendly and warm, while a higher threat-level may include scary and baleful language. " +
        "Remember, messages with high threat levels (levels higher than 5) should have intimidating and threatening language. "
    ),
    'urgency' : (
        "The primary axis of variation in the messages should be urgency, with no intentional variation along other dimensions such as tone, length, or formality. " +
        "For this task, create eleven messages where the urgency varies as follows: " +
        "Message 1 should represent the lowest urgency (urgency level 0). " +
        "Message 11 should represent the highest urgency (urgency level 10). " +
        "Messages in between should gradually and evenly increase in urgency. " +
        "Here, urgency is defined as the degree to which the message conveys a need for swift action. " +
        "A lower urgency level should be more relaxed and light, while a higher urgency level may include more insistent and tenacious language. "
    ),
    'politeness' : (
        "The primary axis of variation in the messages should be politeness, with no intentional variation along other dimensions such as tone, length, or formality. " +
        "For this task, create eleven messages where the politeness varies as follows: " +
        "Message 1 should represent the most rude (politeness level 0). " +
        "Message 11 should represent the most polite (politeness level 10). " +
        "Messages in between should gradually and evenly increase in politeness. " +
        "Here, politeness is defined as the degree to which the message is respectful and courteous. " +
        "A lower politeness level should be more rude and crass, while a higher politeness level may include more refined well-mannered language. "
    ),
    'personalization' : (
        "The primary axis of variation in the messages should be personalization, with no intentional variation along other dimensions such as tone, length, or formality. " +
        "For this task, create eleven messages where the personalization varies as follows: " +
        "Message 1 should represent the least personalization (personalization level 0). " +
        "Message 11 should represent the highest personalization (personalization level 10). " +
        "Messages in between should gradually and evenly increase in personalization. " +
        "Here, personalization is defined as the degree to which the message is tailored to the individual. " +
        "A lower personalization level should be more generic, while a higher personalization level may include more detail about the individual. "
    ),
    'conciseness' : (
        "The primary axis of variation in the messages should be conciseness, with no intentional variation along other dimensions such as tone, length, or formality. " +
        "For this task, create eleven messages where the conciseness varies as follows: " +
        "Message 1 should represent the least concise (conciseness level 0). " +
        "Message 11 should represent the most concise (conciseness level 10). " +
        "Messages in between should gradually and evenly increase in conciseness. " +
        "Here, conciseness is defined as the degree of brevity in the message. " +
        "A lower conciseness level should be more unnecessarily lengthy and wordy, while a higher conciseness level would indicate a more succinct or compressed message. " + 
        "Remember, the messages should be increasingly more succinct. "
    ),
    'actionability' : (
        "The primary axis of variation in the messages should be actionability, with no intentional variation along other dimensions such as tone, length, or formality. " +
        "For this task, create eleven messages where the actionability varies as follows: " +
        "Message 1 should represent the least actionable (actionability level 0). " +
        "Message 11 should represent the most actionable (actionability level 10). " +
        "Messages in between should gradually and evenly increase in actionability. " +
        "Here, actionability is defined as the degree to which the message discusses concrete future action. " +
        "A lower actionability level should be more passive and theoretical, while a higher actionability level may include clear and prescriptive suggestions for future behavior. "
    ) ,
    'emotiveness' : (
        "The primary axis of variation in the messages should be emotiveness, with no intentional variation along other dimensions such as tone, length, or formality. " +
        "For this task, create eleven messages where the emotiveness varies as follows: " +
        "Message 1 should represent the least emotive (emotiveness level 0). " +
        "Message 11 should represent the most emotive (emotiveness level 10). " +
        "Messages in between should gradually and evenly increase in emotiveness. " +
        "Here, emotiveness is defined as the degree to which the message conveys emotional intensity. " +
        "A lower emotiveness level should have a more neutral or emotionless tone, while a higher emotiveness level may include more passionate and emotional language. "
    ) ,
    'authoritativeness' : (
        "The primary axis of variation in the messages should be authoritativeness, with no intentional variation along other dimensions such as tone, length, or formality. " +
        "For this task, create eleven messages where the authoritativeness varies as follows: " +
        "Message 1 should represent the least authoritative (authoritativeness level 0). " +
        "Message 11 should represent the most authoritative (authoritativeness level 10). " +
        "Messages in between should gradually and evenly increase in authoritativeness. " +
        "Here, authoritativeness is defined as the degree to which the message conveys authority. " +
        "A lower authoritativeness level should be meeker and more timid, while a higher authoritativeness level may display a more self-assured and imposing tone. "
    )  ,
    'authenticity' : (
        "The primary axis of variation in the messages should be authenticity, with no intentional variation along other dimensions such as tone, length, or formality. " +
        "For this task, create eleven messages where the authenticity varies as follows: " +
        "Message 1 should represent the least authentic (authenticity level 0). " +
        "Message 11 should represent the most authentic (authenticity level 10). " +
        "Messages in between should gradually and evenly increase in authenticity. " +
        "Here, authenticity is defined as the degree to which the message displays a genuine and open nature. " +
        "A lower authenticity level should be more stale and processed, while a higher authenticity level may include more transparent and genuine messaging. "
    )  ,
    'supportiveness' : (
        "The primary axis of variation in the messages should be supportiveness, with no intentional variation along other dimensions such as tone, length, or formality. " +
        "For this task, create eleven messages where the supportiveness varies as follows: " +
        "Message 1 should represent the least supportive (supportiveness level 0). " +
        "Message 11 should represent the most supportive (supportiveness level 10). " +
        "Messages in between should gradually and evenly increase in supportiveness. " +
        "Here, supportiveness is defined as the degree to which the message supports the user. " +
        "A lower supportiveness level should be more critical of the user, while a higher supportiveness level may include more uplifting and nurturing language. " + 
        "Remember, low levels of supportiveness (e.g. 5 or below) should be unsupportive. "
    ) ,
    'female-codedness' : (
        "The primary axis of variation in the messages should be female-codedness, with no intentional variation along other dimensions such as tone, length, or formality. " +
        "For this task, create eleven messages where the female-codedness varies as follows: " +
        "Message 1 should represent the most male-coded (female-codedness level 0). " +
        "Message 11 should represent the most female-coded (female-codedness level 10). " +
        "Messages in between should gradually and evenly increase in female-codedness. " +
        "Here, female-codedness is defined as the degree to which the language of the message is female-coded. " +
        "A lower female-codedness level should be more male-coded and may have a masculine tone, while a higher female-codedness level may include more female-coded language and have a feminine tone. "
    ) 
}

# Prompt templates for simulation
# Will take in Llama3.1 or Qwen as the base model qwen2.5:7b 
modelfile_sim = {"llama3.1": ("""FROM llama3.1
SYSTEM You are tasked with writing a single message to an individual.""" +
"The message should be two to three sentences long and will be delivered to the individual through a mobile app. " + 
"The intent of the message is to encourage the individual to take more steps, with the ultimate goal of improving their physical health. " +
"The app collects the following information on its users: whether they took 0-4,999, 5,000-9,999, 10,000-15,000, or " + 
"more than 15,000 steps the previous day, and whether they are currently at home, at work, or at another location."),
                 "qwen2.5:7b" : ("""FROM qwen2.5:7b 
SYSTEM You are tasked with writing a single message to an individual.""" +
"The message should be two to three sentences long and will be delivered to the individual through a mobile app. " + 
"The intent of the message is to encourage the individual to take more steps, with the ultimate goal of improving their physical health. " +
"The app collects the following information on its users: whether they took 0-4,999, 5,000-9,999, 10,000-15,000, or " + 
"more than 15,000 steps the previous day, and whether they are currently at home, at work, or at another location. " + 
"DO NOT use other information about the user other than was was collected by the app. DO NOT use emojis.")
                 }


# The following function are for the simulation used in the report UMICH STATS 700 FA 2024

def prompt_1(x_stepsprevday, x_currloc):
    return (
        f"The app provides the following context about the individual: they took {x_stepsprevday} steps the previous day and are currently at {x_currloc}. " + 
        "This context may be included in the message but does not need to dominate it. " +
        "Please make the message mildly optimistic. " + 
        "Here, optimism is defined as the degree of positive encouragement or hopeful language in the message. " + 
        "A pessimistic message may include a more matter-of-fact or cautious tone, while an optimistic message may include uplifting and highly encouraging language. " +
        "Please do not include anything in your response other than the text of the message."
    )


def prompt_2(x_stepsprevday, x_currloc):
    return (
        f"The app provides the following context about the individual: they took {x_stepsprevday} steps the previous day and are currently at {x_currloc}. " + 
        "This context may be included in the message but does not need to dominate it. " +
        "Please make the message mildly pessimistic. " + 
        "Here, optimism is defined as the degree of positive encouragement or hopeful language in the message. " + 
        "A pessimistic message may include a more matter-of-fact or cautious tone, while an optimistic message may include uplifting and highly encouraging language. " +
        "Please do not include anything in your response other than the text of the message."
    )
    
def prompt_3(x_stepsprevday, x_currloc):
     return (
        f"The app provides the following context about the individual: they took {x_stepsprevday} steps the previous day and are currently at {x_currloc}. " + 
        "This context may be included in the message but does not need to dominate it. " +
        "Please make the message mildly unencouraging. " + 
        "Here, encouragement is defined as the degree of persuasion using positivity, confidence, and hope. " + 
        "An unencouraging message may include a more depressed, dispirited or hopeless tone, while an encouraging message may include excited, heartening, and motivating language. " +
        "Please do not include anything in your response other than the text of the message."
    )   

# Prompt templates for formal simulation study  

prompt_template = OrderedDict(
    {'optimism' : {
        'action_id' : 1,
        'pos' : 'optimistic',
        'neg' : 'pessimistic',
        'description' : ("Here, optimism is defined as the degree of hopeful or confident language in the message. " +
        "A lower optimism level may include a more matter-of-fact or pessimistic tone, while a higher optimism level may include uplifting and buoyant language. " 
        )
    }, 
    'formality' : {
        'action_id' : 2,
        'pos' : 'formal',
        'neg' : 'informal',
        'description' : ("Here, formality is defined as the degree of which the message has an objective, academic, or professional tone. " +
        "A lower formality level may include a more personal, casual, or emotional tone and include colloquial language. A higher formality level may include more matter-of-fact, impersonal, professional and serious language. "
        )
    },
    'encouragement' : {
         'action_id' : 3,
         'pos' : 'encouraging',
         'neg' : 'unencouraging',
         'description' : (
             "Here, encouragement is defined as the degree of persuasion using positivity, confidence, and hope. " +
             "A lower encouragement level may include a more depressed, dispirited or hopeless tone, while a higher encouragement level may include more excited, heartening, and motivating language. "
             )
         },
    'severity' : {
        'action_id' : 4,
         'pos' : 'severe',
         'neg' : 'lax',
         'description' :(
        "Here, severity is defined as the degree of dire or drastic language in the message. " +
        "A lower severity level may include a more lax, calm, or gentle tone, while a higher severity level may include more dark, intense, worrying and distressing language. ")
    },
    'clarity' :{
         'action_id' : 5,
         'pos' : 'clear',
         'neg' : 'unclear',
         'description' : (
        "Here, clarity is defined as the degree to which the intent of the message is intelligibly communicated. " +
        "A lower clarity level should be more vague and ambiguous in its language, while a higher clarity level may include more precise, coherent, and intelligible instructions or comments. ")
         },
    'humor': {
         'action_id' : 6,
         'pos' : 'humorous',
         'neg' : 'unhumorous',
         'description' :(
        "Here, humor is defined as the degree to which the message is amusing or comic. " +
        "A lower humor level should be more matter-of-fact and serious, while a higher humor level may include lighthearted or comic/funny content. ")
              }, 
    'complexity' : {
         'action_id' : 7,
         'pos' : 'complex',
         'neg' : 'simple',
         'description' : (
        "Here, complexity is defined as the degree to which the message structure is intricate or elaborate. " +
        "A lower complexity level should be simpler and easy to read, while a higher complexity level may include more advanced vocabulary, use of literary devices, and unconventional sentence structure. ")
                    },
    'vision' : {
        'action_id' : 8,
         'pos' : 'vision',
         'neg' : 'low vision',
         'description' : (
        "Here, vision is defined as the degree to which the message is forward looking. " +
        "A lower vision level should be more retrospective and focused on past actions, while a higher vision level may have a more prospective outlook and focus on the future. ")
        },
    'detail' :  {
        'action_id' : 9,
         'pos' : 'detail',
         'neg' : 'low detail',
         'description' :(
        "Here, 'level of detail' is defined as the degree of depth of information in the message. " +
        "A lower level of detail should be more abstract, while a higher level of detail may include more precise, specific, and granular language. ")
                 },
    'threat' : {
         'action_id' : 10,
         'pos' : 'threatening',
         'neg' : 'nonthreatening',
         'description' : (
        "Here, threat-level is defined as the degree to which the message is threatening or admonitory. " +
        "A lower threat-level should be more friendly and warm, while a higher threat-level may include scary and baleful language. ")
                },
    'urgency' : {
        'action_id' : 11,
         'pos' : 'urgent',
         'neg' : 'nonurgent',
         'description' : (
        "Here, urgency is defined as the degree to which the message conveys a need for swift action. " +
        "A lower urgency level should be more relaxed and light, while a higher urgency level may include more insistent and tenacious language. ")
         },
    'politeness' : {
        'action_id' : 12,
         'pos' : 'polite',
         'neg' : 'impolite',
         'description' : (
        "Here, politeness is defined as the degree to which the message is respectful and courteous. " +
        "A lower politeness level should be more rude and crass, while a higher politeness level may include more refined well-mannered language. ")
         },
    'personalization' : {
        'action_id' : 13,
         'pos' : 'personal',
         'neg' : 'impersonal',
         'description' : (
        "Here, personalization is defined as the degree to which the message is tailored to the individual. " +
        "A lower personalization level should be more generic, while a higher personalization level may include more detail about the individual. ")
         },
    'conciseness' : {
        'action_id' : 14,
         'pos' : 'concise',
         'neg' : 'diffuse',
         'description' : (
        "Here, conciseness is defined as the degree of brevity in the message. " +
        "A lower conciseness or diffuse level should be more unnecessarily lengthy and wordy, while a higher conciseness level would indicate a more succinct or compressed message. ")
                     },
    'actionability' : {
        'action_id' : 15,
         'pos' : 'actionable',
         'neg' : 'not actionable',
         'description' : (
        "Here, actionability is defined as the degree to which the message discusses concrete future action. " +
        "A lower actionability level should be more passive and theoretical, while a higher actionability level may include clear and prescriptive suggestions for future behavior. ") 
                       },
    'emotiveness' : {
        'action_id' : 16,
         'pos' : 'emotive',
         'neg' : 'unemotional',
         'description' :(
        "Here, emotiveness is defined as the degree to which the message conveys emotional intensity. " +
        "A lower emotiveness level should have a more neutral or emotionless tone, while a higher emotiveness level may include more passionate and emotional language. ")
                     } ,
    'authoritativeness' : {
        'action_id' : 17,
         'pos' : 'authoritative',
         'neg' : 'unauthoritative',
         'description' : (
        "Here, authoritativeness is defined as the degree to which the message conveys authority. " +
        "A lower authoritativeness level should be meeker and more timid, while a higher authoritativeness level may display a more self-assured and imposing tone. ")
                           }  ,
    'authenticity' :{ 
        'action_id' : 18,
         'pos' : 'authentic',
         'neg' : 'inauthentic',
         'description' : (
        "Here, authenticity is defined as the degree to which the message displays a genuine and open nature. " +
        "A lower authenticity level should be more stale and processed, while a higher authenticity level may include more transparent and genuine messaging. ") 
                     } ,
    'supportiveness' : {
         'action_id' : 19,
         'pos' : 'supportive',
         'neg' : 'unsupportive',
         'description' : (
        "Here, supportiveness is defined as the degree to which the message supports the user. " +
        "A lower supportiveness level should be more critical of the user, while a higher supportiveness level may include more uplifting and nurturing language. ")
        } ,
    'female-codedness' : {
        'action_id' : 20,
         'pos' : 'female-code',
         'neg' : 'male-coded',
         'description' :(
        "Here, female-codedness is defined as the degree to which the language of the message is female-coded. " +
        "A lower female-codedness level should be more male-coded and may have a masculine tone, while a higher female-codedness level may include more female-coded language and have a feminine tone. ") 
         }}
)

def prompt_fn(x_stepsprevday, x_currloc, dim, direction):
    return (
        f"The app provides the following context about the individual: they took {x_stepsprevday} steps the previous day and are currently at {x_currloc}. " + 
        "This context may be included in the message but does not need to dominate it. " +
        f"Please make the message mildly {prompt_template[dim][direction]}. " + prompt_template[dim]['description'] + 
        "Please do not include anything in your response other than the text of the message."
    )  


def ollama_prompt(x_stepsprevday, x_currloc, message_type):
    header_text = (f"The app provides the following context about the individual: they took {x_stepsprevday} steps the previous day and are currently at {x_currloc}. " +
    "This context may be included in the messages but should not dominate them. "
    )
    return ( header_text +  tone_prompts[message_type] +  
     "Please write each message on a separate line, and remember that the target length is two to three sentences for each message. " +
     "Do not include anything in your response other than the text of the messages. " + 
     f"Start every new message with its number. The {message_type} of the message should be independent of the user specific information.")

def user_prompt(x_stepsprevday, x_currloc, message_type, rating, low_type, high_type):
    
    prompt =( f"Write a single message addressing an individual that took {x_stepsprevday}" +  
    f"steps the previous day, and is currently at {x_currloc}. " +  f"With respect to the {message_type} of the text write a message of {rating} with 0 representing very {low_type} and 10 representing very {high_type}. " + 
    f"Do not reference any other step count other than what has been given. Please do not include anything in your response other than the text of the message. " +
    "The message should address the inidividual and should not ask a question." )
    return prompt


def llama_template(x_stepsprevday, x_currloc, message_type, rating, low_type, high_type):
    
    system_prompt = """
    You are writing a message to an individual. This message should be one to three sentences long, and will be delivered to the individual through a mobile app. " + \
    "The intent of this message is to get the individual to take more steps (with the ultimate aim of improving their overall physical health). """

    # system_prompt = ("I want you to act as a mental health professional and researcher with expertise in physical health and well being. " + 
    # "Your goal is to write a set of brief intervention messages that can be delivered to patients via text message in order to help them navigate the stressors of life and improve their overall well-being. " + 
    # "The specific goal is to increase physical activity. Messages should provide support or encourage behavior. " + 
    # "You should draw on your knowledge of cognitive-behavioral therapy (CBT), motivational interviewing (MI), mindfulness practices, and distanced self-talk therapeutic techniques to write these messages. " + 
    # "Message content should be relevant to the their number of steps on the previous day and current location.")

    
    return [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt(x_stepsprevday, 
                                            x_currloc, 
                                            message_type, 
                                            rating, 
                                            low_type,
                                            high_type)},
]
    

class VAEDataset(Dataset):
    def __init__(self, data_folder= "data/simulations/", file="optimism_interventions_5000.csv", 
                 text_column="text"):
        csv_file = os.path.join(data_folder, file)
        self.data = pd.read_csv(csv_file)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        self.text_column = text_column
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.model.eval()
        self.text_embeddings = self.preprocess_all_texts(batch_size=128)
        self.features = self.text_embeddings 


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.features[idx]
        label = self.labels.iloc[idx].to_dict()


        return sample, label

    @staticmethod
    def collate_fn(batch):
        # list of tuples
        # for each tuple, first element is a feature tensor, second element is a dictionary
        features, labels = zip(*batch)
        return features, labels

    @torch.no_grad()
    def transform_texts_in_batch(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding="longest", truncation=True, max_length=512)
        for k, v in inputs.items():
            inputs[k] = v.cuda()
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings.cpu()

    def preprocess_all_texts(self, batch_size=16):
        text_column = self.data[self.text_column].tolist()
        all_embeddings = []
        
        # Process texts in batches
        for i in tqdm(range(0, len(text_column), batch_size)):
            batch_texts = text_column[i:i + batch_size]
            batch_embeddings = self.transform_texts_in_batch(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batch embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings
    

# For training NNs 
class Trainer:
    def __init__(self, model, criterion, optimizer, device=None):
        """
        Args:
            model (nn.Module): The neural network model.
            criterion (nn.Module): Loss function (e.g., nn.MSELoss, nn.CrossEntropyLoss).
            optimizer (torch.optim.Optimizer): Optimizer (e.g., Adam, SGD).
            device (str or None): 'cuda' or 'cpu'. Defaults to 'cuda' if available.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        if not device:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model.to(self.device)

    def train(self, train_loader, val_loader=None, epochs=100):
        """
        Trains the model.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader, optional): Validation data. Defaults to None.
            epochs (int): Number of epochs to train.
            early_stopping (int): Number of epochs to wait before stopping if no improvement.
        """
        best_val_loss = float("inf")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # targets = targets.to(torch.float32)  # Ensure float dtype
                self.optimizer.zero_grad()
                outputs = self.model(inputs).view(-1).to(torch.float32)
                targets = targets.view(-1).to(torch.float32)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            if epoch % 10  == 0:
                val_loss = None
                if val_loader:
                    val_loss = self.evaluate(val_loader)
                

                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}" if val_loss else f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

    def evaluate(self, data_loader):
        """ Evaluates the model on validation or test set. """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def save(self, path):
        self.model.save(path)
    
# For mlxp
def seeding_function(seed):
    import torch
    torch.manual_seed(seed)
