def scorer(pred: str, gold: str) -> float:
    if gold == pred:
        return 1
    # elif pred in gold:
    #     return 1
    else:
        return 0


def scorer_mgsm(pred: str, gold: str) -> float:
    # pred = pred.replace("$", "")
    # gold = gold.replace("$", "")
    # return float(pred.startswith(gold))
    if gold == pred:
        return 1
    else:
        return 0


# see metrics in BOT
def scorer_aqua(pred: str, gold: str) -> float:
    # pred = pred.replace("$", "")
    # gold = gold.replace("$", "")
    # return float(pred.startswith(gold))
    if gold == pred:
        return 1
    else:
        return 0


def scorer_swamp(pred: str, gold: str) -> float:
    if pred.strip() == "":
        return 0
    if gold.strip() == pred.strip() or pred.strip()+".0" == gold.strip():
        return 1
    else:
        return 0


def scorer_bamboogle(pred: str, gold: str) -> float:
    # if gold.upper() == pred.upper():
    #     return 1
    # elif pred.upper() in gold.upper():
    #     return 1
    # else:
    #     return 0
    if pred.lower() == gold.lower() \
            or pred.lower().split(' ')[0] in gold.lower() \
            or gold.lower().split(' ')[0] in pred.lower() \
            or pred.lower().startswith(gold.lower()):
        return 1
    else:
        return 0


def scorer_strategyqa(pred: str, gold: str) -> float:
    if pred.strip() == "":
        return 0
    if gold.strip() == pred.strip():
        return 1
    else:
        return 0
def scorer_date(pred: str, gold: str) -> float:
    from datetime import datetime

    if pred.strip() == "" or pred == "None":
        return 0
    else:
        try:
            date1 = datetime.strptime(pred, '%d/%m/%Y')
            date2 = datetime.strptime(gold, '%d/%m/%Y')
            if date1 == date2:
                return 1
            else:
                return 0
        except:
            return 0
    if gold.strip() == pred.strip():
        return 1
    else:
        return 0


def scorer_sports(pred: str, gold: str) -> float:
    # if pred.strip() == "":
    #     return 0
    # if gold.strip() == pred.strip():
    #     return 1
    # else:
    #     return 0
    if "yes" in pred.lower() and "yes" in gold.lower():
        return 1
    elif "no" in pred.lower() and "no" in gold.lower():
        return 1
    else:
        return 0


def scorer_coinflip(pred: str, gold: str) -> float:
    # if pred.strip() == "":
    #     return 0
    # if gold.strip() == pred.strip():
    #     return 1
    # else:
    #     return 0
    if "yes" in pred.lower() and "yes" in gold.lower():
        return 1
    elif "no" in pred.lower() and "no" in gold.lower():
        return 1
    else:
        return 0


def scorer_lastletters(pred: str, gold: str) -> float:
    if pred.strip() == "":
        return 0
    if gold.strip() == pred.strip():
        return 1
    else:
        return 0
