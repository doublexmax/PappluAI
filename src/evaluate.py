from collections import defaultdict
from functools import reduce, cached_property as __cached__

"""
First: calculate the number of pure sequences in the hand.
if the number of pure sequences is less than 3, return False.

Second: find the number of unmatched cards in the hand.
if the number of unmatched cards is 0 return True

third: find the number of extra jokers (outside of the sequences)

fourth: match the remaining cards outside of jokers to sets

now we are left with 1) random cards 2) cards that are almost sequences 3) cards that are almost sets 4) jokers

go over sets that have two cards:
- if there is a card that can be removed from a seqeunce to make a set, do it
- if there is a joker, use it to make a set

go over sequences that have two cards:
- apply a joker

final: return True if no cards left

"""

def generate_sequences(starting_hand):
    hands = []

    def helper(i, hand, num_padding_sequences = 0, initial_run = 0):
        idx = i
        running = initial_run
        num_sequence = num_padding_sequences
        while idx < 52:
            if hand[idx] == 0: # no card present
                running = 0
            else:
                running += 1
            
            if running >= 3:
                helper(idx+1, hand.copy(), num_padding_sequences=num_sequence, initial_run=running)
                
                for k in range(running):
                    hand[idx - k] -= 1
                idx -= running - 1
                num_sequence += 1

                hands.append((hand.copy(), num_sequence))

                # helper(idx, hand, num_padding_sequences=num_padding_sequences)

                running = 0
            else:
                idx += 1

    helper(0, starting_hand.copy())

    final_hands = defaultdict(int)

    # since we can end up with the same hand after different types of matches, we only care about the max number of sequences per resulting hand
    for hand, num_sequences in hands:
        print(hand, num_sequences, ''.join(str(x) for x in hand))
        final_hands[''.join(str(x) for x in hand)] = max(num_sequences, final_hands[''.join(str(x) for x in hand)])

    return [(list(int(x) for x in hand), num_sequences) for hand, num_sequences in final_hands.items()]

def generate_pure_sets(starting_hand):
    hands = []

    def helper(i, hand, num_padding_sets = 0):
        idx = i
        num_sets = num_padding_sets

        while idx < 52:
            if hand[idx] == 3:
                helper(idx+1, hand.copy(), num_padding_sets=num_sets)
                
                hand[idx] -= 3
                num_sets += 1

                hands.append((hand.copy(), num_sets))

                # helper(idx, hand, num_padding_sequences=num_padding_sequences)
            idx += 1

    helper(0, starting_hand.copy())

    final_hands = defaultdict(int)

    # since we can end up with the same hand after different types of matches, we only care about the max number of sequences per resulting hand
    for hand, num_sequences in hands:
        final_hands[''.join(str(x) for x in hand)] = max(num_sequences, final_hands[''.join(str(x) for x in hand)])

    return [(list(int(x) for x in hand), num_sequences) for hand, num_sequences in final_hands.items()]

def match_cards(starting_hand):
    sequence_hands = generate_sequences(starting_hand.copy())

    hands = []
    hands.extend(sequence_hands)

    for hand, num_sequences in sequence_hands:
        set_hands = generate_pure_sets(hand.copy())
        for hand, num_sets in set_hands:
            hands.append((hand, num_sequences + num_sets))
    
    final_hands = defaultdict(int)

    # since we can end up with the same hand after different types of matches, we only care about the max number of sequences per resulting hand
    for hand, num_sequences in hands:
        final_hands[''.join(str(x) for x in hand)] = max(num_sequences, final_hands[''.join(str(x) for x in hand)])

    return [(list(int(x) for x in hand), num_sequences) for hand, num_sequences in final_hands.items()]

def match_sets(starting_hand):
    print(starting_hand)

    hands = [starting_hand.copy()]

    def helper(i, hand):
        while i < 13:
            if hand[i] and hand[i+13] and hand[i+26] and hand[i+39]:
                helper(i+1, hand.copy())
                hand[i] -= 1
                hand[i+13] -= 1
                hand[i+26] -= 1
                hand[i+39] -= 1
                hands.append(hand.copy())

            elif starting_hand[i] and starting_hand[i+13] and starting_hand[i+26]:
                helper(i+1, hand.copy())
                hand[i] -= 1
                hand[i+13] -= 1
                hand[i+26] -= 1
                hands.append(hand.copy())

            else:
                i += 1
    
    helper(0, starting_hand.copy())
    
    return hands

def filter_jokers(hand, joker):
    return sum(hand) - hand[joker]

def match_joker_to_sequences(hand, joker):
    hands = [hand.copy()]

    num_jokers = hand[joker]

    def helper(i, hand, num_jokers, initial_run = 0):
        idx = i
        running = initial_run
        temp_num_jokers = num_jokers
    
        while idx < 52:
            if hand[idx]:
                running += 1
            elif hand[idx] == 0 and temp_num_jokers: # no card present but joker available
                running += 1
                temp_num_jokers -= 1
            else:
                running = 0
                temp_num_jokers = num_jokers
            
            if running >= 3:
                helper(idx+1, hand.copy(), temp_num_jokers, initial_run=running)
                
                for k in range(running):
                    hand[idx - k] -= 1
                idx -= running - 1
                num_jokers = temp_num_jokers

                hands.append(hand.copy())

                # helper(idx, hand, num_padding_sequences=num_padding_sequences)

                running = 0
            else:
                idx += 1
        
    helper(0, hand.copy(), num_jokers)

    return hands

def match_joker_to_sets(hand, joker):
    hands = [hand.copy()]

    num_jokers = hand[joker]

    def helper(i, hand, num_jokers):
        temp_num_jokers = num_jokers
        while i < 13:
            if cur_sum := max(hand[i],1) + max(hand[i+13],1) + max(hand[i+26],1) + max(hand[i+39],1) + temp_num_jokers >= 4:
                helper(i+1, hand.copy(), temp_num_jokers)
                hand[i] -= 1 if hand[i] else 0
                hand[i+13] -= 1 if hand[i+13] else 0
                hand[i+26] -= 1 if hand[i+26] else 0
                hand[i+39] -= 1 if hand[i+39] else 0

                temp_num_jokers -= 4 - cur_sum

                hands.append(hand.copy())

            elif cur_sum := max(hand[i],1) + max(hand[i+13],1) + max(hand[i+26],1) + temp_num_jokers >= 4:
                helper(i+1, hand.copy())
                hand[i] -= 1 if hand[i] else 0
                hand[i+13] -= 1 if hand[i+13] else 0
                hand[i+26] -= 1 if hand[i+26] else 0

                temp_num_jokers -= 3 - cur_sum

                hands.append(hand.copy())

            else:
                i += 1
    
    helper(0, hand.copy(), num_jokers)

    return hands

def apply_joker(hand, joker):
    after_sequences = match_joker_to_sequences(hand.copy(), joker)

    after_sets = []

    for hand in after_sequences:
        after_sets.extend(match_joker_to_sets(hand.copy(), joker))
    print(after_sets, 'after sets apply joker')
    final_hand = reduce(lambda x, y: x if sum(x) - x[joker] <= sum(y) - y[joker] else y, after_sets)

    return final_hand

def matched_rate(hand, joker, required_sequences = 3, cards_in_hand = 21):
    """
    Determine if a hand satisfies the condition:
        - At least *required_sequences* pure sequences.
        - The remaining sequences can be impure sequences or sets. 
    """

    matched_hands = match_cards(hand.copy())

    possible_hands = [hand for hand, num_sequences in matched_hands if num_sequences >= required_sequences]

    # check if there are any possible hands
    if not possible_hands:
        print('no possible hands')
        return cards_in_hand
    
    print(possible_hands, 'possible hands')

    # check after matching to sequences
    if any(sum(x) - x[joker] == 0 for x in possible_hands):
        print('matched by sequences')
        return 0 

    after_sets = [match_sets(hand)[0] for hand in possible_hands]

    print(after_sets, 'after sets')

    # check after matching to sets
    if any(sum(x) - x[joker] == 0 for x in after_sets):
        print('matched using sets')
        return 0 
    
    # check after trying to use joker

    best_hand = 0

    for hand in possible_hands:
        after_joker = apply_joker(hand.copy(), joker)
        print(after_joker, 'after joker final')
        if cur_marks:=sum(after_joker) - after_joker[joker] <= best_hand:
            best_hand = cur_marks
    
    return best_hand

hand = [2]*7 + [1] + [0] + [1] + [0]*42
x = generate_sequences(hand.copy())
y = generate_pure_sets(hand.copy())
#print(x)
#print(y)
print(matched_rate(hand.copy(), 25))