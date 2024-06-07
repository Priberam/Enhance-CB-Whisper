import re
import unicodedata
from collections import namedtuple

Token = namedtuple("Token", ["index", "start", "end", "text", "type"])


class PriberamTokenizer:
    def __init__(self):
        flags = re.UNICODE | re.MULTILINE
        self.groups = ["abbr", "alphanum", "newline", "space", "punctuation", "full_stop"]
        # The abbr works by finding words that have between 1 and 3 letters and are followed by a
        # ".", for instance D. José, Dr. Ribeiro, Dra. Ribeiro, E. U. A., etc. The downside is that
        # it also matches any final word in a sentence that has at most 3 letters. Thus, it can
        # lead to some longer sentences in those corner cases.
        # Do not use \s if you don't want to also match \r and \n. Python regex engine does not
        # support \h, which only catches horizontal white spaces.
        # AMM : removed the abbr the model should be able to handle that by itself
        # Use: [^\S\r\n], following: https://stackoverflow.com/a/17752989
        punctuation_chars = ["\\" + chr(i) for i in range(0x10000) if unicodedata.category(chr(i)).startswith('P')]
        self.regex = re.compile(
            #"(?P<abbr>[\w]{1,3}\.[^\S\r\n])|(?P<alphanum>[\w]+)|(?P<newline>[\r\n]+)|(?P<space>[^\S\r\n\u00a0]+)|(?P<full_stop>[\.][^\S\r\n])|(?P<punctuation>[\!\.\,\?\:\;\(\)\-\\""\{])",
             f"(?P<alphanum>[\w]+)|(?P<newline>[\r\n]+)|(?P<space>[\s \u00a0]+)|(?P<full_stop>([\.] )|。|።)|(?P<punctuation>[{''.join(punctuation_chars)}])",
            #"(?P<alphanum>[\w]+)|(?P<newline>[\r\n]+)|(?P<space>[^\S\s\r\n\u00a0]+)|(?P<full_stop>[\.][^\S\r\n])|(?P<punctuation>[\!\.\,\?\:\;\(\)\-\\\"\"\{])",
            flags=flags,
        )
        self.new_line = re.compile("[\r\n]+", flags=flags)
        #self.Token = namedtuple("Token", ["index", "start", "end", "text", "type"])

    def is_nonlatin_fullstop(self, char):
        return char in  ["。", "።"]

    def tokenize(self, text):

        document_sentences = []
        sentence = []
        pos = 0
        index = -1

        regex_matches = re.finditer(self.regex, text)

        for match in regex_matches:

            match_text = match.group()

            # Deal with UNKs:
            # When pos < match.start() it means some part of the string was missed
            if pos < match.start():
                if not sentence:
                    index = 0
                    sentence = []
                    document_sentences.append(sentence)
                # Add new token
                index += 1
                #token = self.Token(
                token = Token(
                    index=index,
                    start=pos,
                    end=pos + len(text[pos : match.start()]),
                    text=text[pos : match.start()],
                    type="UNK",
                )
                document_sentences[-1].append(token)
                pos = match.start()

            # Assert we are only matching one group
            if len([g for g in match.groupdict().values() if g]) != 1:
                raise Exception("Entity is matching multiple groups")

            # Define new type based upon the retrieved name (group)
            name = match.lastgroup
            if name == "alphanum":
                text_type = "text"
            elif name == "abbr":
                text_type = "text"
            elif name == "newline":
                text_type = "paragraph"
            elif name == "space":
                text_type = "space"
            elif name == "punctuation":
                text_type = "punctuation"
            elif name == "full_stop":
                text_type = "full_stop"

            # Deal with paragraphs:
            # Match multiple consecutive line breaks
            if text_type == "paragraph":
                new_line_matches = re.finditer(self.new_line, match_text)
                for new_line_match in new_line_matches:
                    new_line_match_text = new_line_match.group()
                    if not sentence:
                        index = 0
                        sentence = []
                        document_sentences.append(sentence)
                    # Add new token
                    index += 1
                    #token = self.Token(
                    token = Token(
                        index=index,
                        start=pos + new_line_match.start(),
                        end=pos + new_line_match.start() + len(new_line_match_text),
                        text=new_line_match_text,
                        type="paragraph",
                    )
                    document_sentences[-1].append(token)
                sentence = None
            # If not paragraph add new token
            else:
                if not sentence:
                    index = 0
                    sentence = []
                    document_sentences.append(sentence)
                # Add new token
                index += 1
                #token = self.Token(
                token = Token(
                    index=index,
                    start=match.start(),
                    end=match.start() + len(match_text),
                    text=match_text,
                    type=text_type,
                )
                document_sentences[-1].append(token)

                # Deal with full stop:
                # Set sentence to None, so that the next time we add a new token,
                # we append a new/empty sentence to the document_sentences list
                # and start appending new tokens to that "clean" sentence
                if text_type == "full_stop" and ((self.is_nonlatin_fullstop(match_text)) or 
                                                 ((len(document_sentences[-1]) > 2 and 
                                                   len(document_sentences[-1][-2].text) > 2))):
                    sentence = None

            pos = match.end()

        # Deal with possible UNKs at the end of the text.
        if pos < len(text):
            if not sentence:
                sentence = []
                document_sentences.append(sentence)
            # Add new token
            index += 1
            #token = self.Token(
            token = Token(
                index=index,
                start=pos,
                end=pos + len(text[pos:]),
                text=text[pos:],
                type="UNK",
            )
            document_sentences[-1].append(token)

        return document_sentences
        
    def just_split_sentences(self, text):
        document_sentences = []
        sentences = self.tokenize(text)
        for sent_info in sentences:
            sentence = []
            #token = self.Token(
            token = Token(
                index=0,
                start=sent_info[0].start,
                end=sent_info[-1].end,
                text=text[sent_info[0].start:sent_info[-1].end],
                type="UNK",
                    )
            sentence.append(token)
            document_sentences.append(sentence)

        #if len(sentence) > 5:
         #   sentences.append(sentence)  
        return document_sentences