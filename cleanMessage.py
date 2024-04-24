def clean_message(message):
    message = message.replace(".", "")
    message = message.replace(",", "")
    message = message.replace(";", "")
    message = message.replace("!", "")
    message = message.replace("?", "")
    message = message.replace("(", "")
    message = message.replace(")", "")
    message = message.replace("\\", "")
    message = message.replace("\"", "")

    # remove discord effects
    message = message.replace("*", "")
    message = message.replace("_", "")
    message = message.replace("~", "")
    message = message.replace("`", "")
    message = message.replace(">", "")
    message = message.replace("<", "")
    message = message.replace("||", "")
    message = message.replace("```", "")
    message = message.replace("~~", "")
    message = message.replace(":", "")
    message = message.replace("#", "")
    message = message.replace("@", "")


    # remove stopwords
    message = message.lower()
    message = ' '.join([word for word in message.split() if word not in STOP_WORDS or word in USER_NAMES])

    return message