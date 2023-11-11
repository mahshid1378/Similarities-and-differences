import urllib
import urllib2
import hashlib
import random
import email
import email.message
import email.encoders
import os
import sys

def submit(partId):
    print ("==\n== [nlp-class] Submitting Solutions | Programming Exercise %s\n==" % homework_id())
    if(not partId):
        partId = promptPart()

    partNames = validParts()
    if not isValidPartId(partId):
        print ('!! Invalid homework part selected.')
        print ('!! Expected an integer from 1 to %d.' % (len(partNames) + 1))
        print ('!! Submission Cancelled')
        return

    (login, password) = loginPrompt()
    if not login:
        print ('!! Submission Cancelled')
        return
    print ('\n== Connecting to nlp-class ... ')
    if partId == len(partNames) + 1:
        submitParts = range(1, len(partNames) + 1)
    else:
        submitParts = [partId]
    for partId in submitParts:
        (login, ch, state, ch_aux) = getChallenge(login, partId)
        if((not login) or (not ch) or (not state)):
            print ('\n!! Error: %s\n' % login)
            return
        ch_resp = challengeResponse(login, password, ch)
        (result, string) = submitSolution(login, ch_resp, partId, \
                output(partId, ch_aux), source(partId), state, ch_aux)
        print ('\n== [nlp-class] Submitted Homework %s - Part %d - %s' % \
                (homework_id(), partId, partNames[partId - 1]))
        print ('== %s' % string.strip())

def promptPart():
    print('== Select which part(s) to submit: ' + homework_id())
    partNames = validParts()
    srcFiles = sources()
    for i in range(1,len(partNames)+1):
        print ('==   %d) %s [ %s ]' % (i, partNames[i - 1], srcFiles[i - 1]))
    print ('==   %d) All of the above \n==\nEnter your choice [1-%d]: ' % \
            (len(partNames) + 1, len(partNames) + 1))
    selPart = raw_input('')
    partId = int(selPart)
    if not isValidPartId(partId):
        partId = -1
    return partId

def validParts():
    partNames = [ 'Inverted Index Dev', \
                  'Inverted Index Test', \
                  'Boolean Retrieval Dev', \
                  'Boolean Retrieval Test', \
                  'TF-IDF Dev', \
                  'TF-IDF Test', \
                  'Cosine Similarity Dev', \
                  'Cosine SImilarity Test'
                ]
    return partNames

def sources():
    srcs = [ [ 'IRSystem.py'], \
             [ 'IRSystem.py'], \
             [ 'IRSystem.py'], \
             [ 'IRSystem.py'], \
             [ 'IRSystem.py'], \
             [ 'IRSystem.py'], \
             [ 'IRSystem.py'], \
             [ 'IRSystem.py'] \
           ]
    return srcs

def isValidPartId(partId):
    partNames = validParts()
    return (partId and (partId >= 1) and (partId <= len(partNames) + 1))

def loginPrompt():
    (login, password) = basicPrompt()
    return login, password

def basicPrompt():
    login = raw_input('Login (Email address): ')
    password = raw_input('Password: ')
    return login, password

def homework_id():
    return '7'

def getChallenge(email, partId):
    url = challenge_url()
    values = {'email_address' : email, \
              'assignment_part_sid' : "%s-%d" % (homework_id(), partId), \
              'response_encoding' : 'delim'}
    data = urllib.urlencode(values)
    req = urllib2.Request(url, data)
    response = urllib2.urlopen(req)
    text = response.read().strip()
    splits = text.split('|')
    if(len(splits) != 9):
        print ('Badly formatted challenge response: %s' % text)
        return None
    return (splits[2], splits[4], splits[6], splits[8])

def challengeResponse(email, passwd, challenge):
    sha1 = hashlib.sha1()
    sha1.update("".join([challenge, passwd])) 
    digest = sha1.hexdigest()
    strAnswer = ''
    for i in range(0, len(digest)):
        strAnswer = strAnswer + digest[i]
    return strAnswer
    
def challenge_url():
    return 'https://class.coursera.org/nlp/assignment/challenge'

def submit_url():
    return 'https://class.coursera.org/nlp/assignment/submit'

def submitSolution(email_address, ch_resp, part, output, source, state, ch_aux):
    source_64_msg = email.message.Message()
    source_64_msg.set_payload(source)
    email.encoders.encode_base64(source_64_msg)
    output_64_msg = email.message.Message()
    output_64_msg.set_payload(output)
    email.encoders.encode_base64(output_64_msg)
    values = { 'assignment_part_sid' : ("%s-%d" % (homework_id(), part)), \
               'email_address' : email_address, \
               'submission' : output_64_msg.get_payload(), \
               'submission_aux' : source_64_msg.get_payload(), \
               'challenge_response' : ch_resp, \
               'state' : state \
           }
    url = submit_url()
    data = urllib.urlencode(values)
    req = urllib2.Request(url, data)
    response = urllib2.urlopen(req)
    string = response.read().strip()
    result = 0
    return result, string

def source(partId):
    src = ''
    src_files = sources()
    if partId <= len(src_files):
        flist = src_files[partId - 1]
        for fname in flist:
            f = open(fname)
            src = src + f.read()
            f.close()
            src = src + '||||||||'
    return src

from IRSystem import IRSystem
def output(partId, ch_aux):
    version = 1
    output = [partId, version]
    irsys = IRSystem()
    irsys.read_data('C:/Users/hp 850/Desktop/Data/RiderHaggard')
    irsys.index()
    irsys.compute_tfidf()
    out = sys.stdout
    if partId in [2,4,6,8]:   
        sys.stdout = open(os.devnull, 'w')
    if partId == 1 or partId == 2:
        queries = ch_aux.split(", ")
        for query in queries:
            posting = irsys.get_posting_unstemmed(query)
            output.append(list(posting))
    elif partId == 3 or partId == 4:
        queries = ch_aux.split(", ")
        for query in queries:
            result = irsys.query_retrieve(query)
            result = list(result)
            output.append(result)
    elif partId == 5 or partId == 6:
        queries = ch_aux.split("; ")
        for query in queries:
            word, docID = query.split(", ")
            result = irsys.get_tfidf_unstemmed(word, int(docID));
            output.append(result)
    elif partId == 7 or partId == 8:
        queries = ch_aux.split(", ")
        for query in queries:
            results = irsys.query_rank(query)
            first_result = [results[0][0], results[0][1]]
            output.append(first_result)
    else:
        print ("Unknown partId: %d" % partId)
        return None
    if partId in [2,4,6,8]:   
        sys.stdout = out
    output = str(output)
    return output
submit(0)