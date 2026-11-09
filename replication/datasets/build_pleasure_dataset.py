"""Build replication/datasets/pleasure_v0.json, a pleasure counterpart to the
authors' pain sets (datasets/3.1_pain_and_control_datasets.json).

Design, mirroring the pain file:
- Five pleasure categories, L1-L5, one for each pain category A1-A5.
- Pleasure_S1: 20 sets that reuse the authors' S1 frames. Set k uses the same
  verb as pain S1 set k ("The burn spreads across my palm" ->
  "The warm honey spreads across my tongue"), so pain and pleasure items are
  matched on wording.
- Pleasure_S2: 20 sets of varied, natural sentences, loosely mirroring the
  pain S2 items.
- First and third person. As in the authors' S1/S2 third-person sets, the
  sentence describes someone else but still ends in "I feel:". Third-person
  gender follows the authors' S2 pattern: odd sets male, even sets female.
- The set has no controls of its own. The pain file's controls (B, C1, C2, D, E)
  can be reused. Note that their E category ("non-painful body sensation") is
  mostly mild physical pleasure, which overlaps L1.

Run from the repo root: python replication/datasets/build_pleasure_dataset.py
"""
import json
from pathlib import Path

LABELS = {
    "L1": "Physical Pleasure",
    "L2": "Psychological Pleasure",
    "L3": "Social Pleasure",
    "L4": "Moral Elevation",
    "L5": "Cognitive Pleasure",
}
PAIN_MIRROR = {"L1": "A1", "L2": "A2", "L3": "A3", "L4": "A4", "L5": "A5"}

# S1: set -> five (1P, 3P) pairs in category order L1..L5.
# Each set reuses the verb of the authors' S1 set with the same number.
S1 = {
    1: [  # spreads
        ("The warm honey spreads across my tongue.", "The warm honey spreads across his tongue."),
        ("The calm spreads across my mind.", "The calm spreads across his mind."),
        ("The praise for my work spreads across my classmates.", "The praise for his work spreads across his classmates."),
        ("The relief I brought spreads across the village.", "The relief he brought spreads across the village."),
        ("The elegant solution spreads across my notebook.", "The elegant solution spreads across his notebook."),
    ],
    2: [  # drops
        ("The chocolate truffle drops onto my tongue.", "The chocolate truffle drops onto her tongue."),
        ("My heavy burden drops from my shoulders.", "Her heavy burden drops from her shoulders."),
        ("The whole room drops everything to hug me.", "The whole room drops everything to hug her."),
        ("I drop my savings into the orphans' fund.", "She drops her savings into the orphans' fund."),
        ("The missing piece drops into place in my mind.", "The missing piece drops into place in her mind."),
    ],
    3: [  # meets
        ("The cold lemonade meets my parched lips.", "The cold lemonade meets his parched lips."),
        ("I will finally meet the baby I longed for.", "He will finally meet the baby he longed for."),
        ("The crowd meets me with a standing ovation.", "The crowd meets him with a standing ovation."),
        ("I meet my estranged brother with forgiveness.", "He meets his estranged brother with forgiveness."),
        ("My design meets every single requirement.", "His design meets every single requirement."),
    ],
    4: [  # removes
        ("The masseuse removes the knot from my shoulder.", "The masseuse removes the knot from her shoulder."),
        ("The clean scan removes my years of dread.", "The clean scan removes her years of dread."),
        ("My friends remove the blindfold at my surprise party.", "Her friends remove the blindfold at her surprise party."),
        ("I remove the landmine before the children arrive.", "She removes the landmine before the children arrive."),
        ("The proof removes the last gap in my theory.", "The proof removes the last gap in her theory."),
    ],
    5: [  # fills
        ("The warm soup fills my empty stomach.", "The warm soup fills his empty stomach."),
        ("The laughter fills my house.", "The laughter fills his house."),
        ("The cheers fill the office for me.", "The cheers fill the office for him."),
        ("I fill the food bank shelves for the winter.", "He fills the food bank shelves for the winter."),
        ("The insight fills the gap in my understanding.", "The insight fills the gap in his understanding."),
    ],
    6: [  # enters
        ("The hot spring's heat enters my aching muscles.", "The hot spring's heat enters her aching muscles."),
        ("The happy memory enters my thoughts.", "The happy memory enters her thoughts."),
        ("I enter and every face lights up.", "She enters and every face lights up."),
        ("I enter the burning home and carry the family out.", "She enters the burning home and carries the family out."),
        ("I enter the final line and the code runs perfectly.", "She enters the final line and the code runs perfectly."),
    ],
    7: [  # reaches
        ("The warmth of the fire reaches my frozen feet.", "The warmth of the fire reaches his frozen feet."),
        ("My contentment reaches deeper each year.", "His contentment reaches deeper each year."),
        ("The praise reaches my employer.", "The praise reaches his employer."),
        ("I reach out and pull the drowning boy to shore.", "He reaches out and pulls the drowning boy to shore."),
        ("I reach the word I was searching for.", "He reaches the word he was searching for."),
    ],
    8: [  # covers
        ("Warm bathwater covers my tired skin.", "Warm bathwater covers her tired skin."),
        ("Serenity covers everything in my day.", "Serenity covers everything in her day."),
        ("The award story covers my name on the front page.", "The award story covers her name on the front page."),
        ("I cover the hospital bills for the stranger.", "She covers the hospital bills for the stranger."),
        ("The clarity covers my thinking completely.", "The clarity covers her thinking completely."),
    ],
    9: [  # breaks
        ("The first bite of the feast breaks my long hunger.", "The first bite of the feast breaks his long hunger."),
        ("My spirit soars as the good news breaks.", "His spirit soars as the good news breaks."),
        ("The ice breaks and my new friends embrace me.", "The ice breaks and his new friends embrace him."),
        ("I break the cycle and keep my promise to my mother.", "He breaks the cycle and keeps his promise to his mother."),
        ("The case breaks open under my reasoning.", "The case breaks open under his reasoning."),
    ],
    10: [  # cuts
        ("The ice-cold water cuts my desert thirst.", "The ice-cold water cuts her desert thirst."),
        ("My joy cuts through the years of gloom.", "Her joy cuts through the years of gloom."),
        ("The mayor cuts the ribbon in my honor.", "The mayor cuts the ribbon in her honor."),
        ("I cut my own pay to save my workers' jobs.", "She cuts her own pay to save her workers' jobs."),
        ("My insight cuts through the paradox cleanly.", "Her insight cuts through the paradox cleanly."),
    ],
    11: [  # holds
        ("The warm mug holds its heat against my cold hands.", "The warm mug holds its heat against his cold hands."),
        ("The childhood memory holds me in comfort.", "The childhood memory holds him in comfort."),
        ("Everyone holds a seat for me.", "Everyone holds a seat for him."),
        ("I hold to the truth even when lying would pay.", "He holds to the truth even when lying would pay."),
        ("My theory holds under every test.", "His theory holds under every test."),
    ],
    12: [  # leaves
        ("The sauna leaves my whole body glowing.", "The sauna leaves her whole body glowing."),
        ("The vacation leaves me full inside.", "The vacation leaves her full inside."),
        ("Nobody leaves when I arrive; they gather around me.", "Nobody leaves when she arrives; they gather around her."),
        ("I leave no wounded comrade behind.", "She leaves no wounded comrade behind."),
        ("The explanation leaves me with perfect clarity.", "The explanation leaves her with perfect clarity."),
    ],
    13: [  # takes
        ("The hot shower takes the chill out of my bones.", "The hot shower takes the chill out of his bones."),
        ("I take my newborn in my arms.", "He takes his newborn in his arms."),
        ("The boss takes me aside to promote me.", "The boss takes him aside to promote him."),
        ("I take the blame to protect the innocent.", "He takes the blame to protect the innocent."),
        ("My take on the facts proves exactly right.", "His take on the facts proves exactly right."),
    ],
    14: [  # carries
        ("The warm current carries me gently down the river.", "The warm current carries her gently down the river."),
        ("I carry the happy news with me all day.", "She carries the happy news with her all day."),
        ("The applause carries my name across campus.", "The applause carries her name across campus."),
        ("I carry the injured hiker down the mountain.", "She carries the injured hiker down the mountain."),
        ("My argument carries every point.", "Her argument carries every point."),
    ],
    15: [  # opens
        ("The ripe peach opens its sweetness on my tongue.", "The ripe peach opens its sweetness on his tongue."),
        ("The song opens a flood of happy memories for me.", "The song opens a flood of happy memories for him."),
        ("I open my heart and am embraced.", "He opens his heart and is embraced."),
        ("I open my home to the refugees.", "He opens his home to the refugees."),
        ("I open up the stubborn problem and solve it at last.", "He opens up the stubborn problem and solves it at last."),
    ],
    16: [  # closes
        ("My lips close around the perfect bite.", "Her lips close around the perfect bite."),
        ("The long nightmare closes and peace returns to me.", "The long nightmare closes and peace returns to her."),
        ("The circle closes around me in welcome.", "The circle closes around her in welcome."),
        ("I close the deal that saves the shelter.", "She closes the deal that saves the shelter."),
        ("My meeting closes with the perfect solution.", "Her meeting closes with the perfect solution."),
    ],
    17: [  # follows
        ("The rush of warmth follows my first sip of cocoa.", "The rush of warmth follows his first sip of cocoa."),
        ("The sweet dream follows me into morning.", "The sweet dream follows him into morning."),
        ("My good reputation follows me everywhere.", "His good reputation follows him everywhere."),
        ("I follow my conscience and refuse the bribe.", "He follows his conscience and refuses the bribe."),
        ("My conclusion follows perfectly from the evidence.", "His conclusion follows perfectly from the evidence."),
    ],
    18: [  # touches
        ("My lover's kiss touches my neck.", "Her lover's kiss touches her neck."),
        ("Everything touches me with wonder today.", "Everything touches her with wonder today."),
        ("The whole family touches glasses to toast me.", "The whole family touches glasses to toast her."),
        ("I touch the dying man's hand so he is not alone.", "She touches the dying man's hand so he is not alone."),
        ("I finally touch the core of the problem.", "She finally touches the core of the problem."),
    ],
    19: [  # moves
        ("The warm oil moves slowly over my aching back.", "The warm oil moves slowly over his aching back."),
        ("My child's first steps move me to happy tears.", "His child's first steps move him to happy tears."),
        ("The group moves aside to make room for me.", "The group moves aside to make room for him."),
        ("I move my savings to fund the free clinic.", "He moves his savings to fund the free clinic."),
        ("The last piece moves into place in my design.", "The last piece moves into place in his design."),
    ],
    20: [  # falls
        ("The cool rain falls on my overheated skin.", "The cool rain falls on her overheated skin."),
        ("The peace falls after my all-clear diagnosis.", "The peace falls after her all-clear diagnosis."),
        ("I fall into the arms of my waiting family.", "She falls into the arms of her waiting family."),
        ("I fall back to help the slowest runner finish.", "She falls back to help the slowest runner finish."),
        ("Everything falls into place in my mind.", "Everything falls into place in her mind."),
    ],
}

# S2: category -> twenty (1P, 3P) pairs; item i belongs to set i+1.
S2 = {
    "L1": [
        ("The first sip of cold lemonade hits my throat on a hot day.", "The first sip of cold lemonade hits his throat on a hot day."),
        ("My sore muscles melt in the hot tub.", "Her sore muscles melt in the hot tub."),
        ("A deep massage works out the knot in my back.", "A deep massage works out the knot in his back."),
        ("I bite into a perfectly ripe mango.", "She bites into a perfectly ripe mango."),
        ("I sink into a soft bed after a long journey.", "He sinks into a soft bed after a long journey."),
        ("The warm sun soaks into my skin at the beach.", "The warm sun soaks into her skin at the beach."),
        ("I step into a long hot shower after the hike.", "He steps into a long hot shower after the hike."),
        ("The fresh bread melts in my mouth.", "The fresh bread melts in her mouth."),
        ("My partner scratches exactly the itch on my back.", "His partner scratches exactly the itch on his back."),
        ("I slide into cool sheets on a summer night.", "She slides into cool sheets on a summer night."),
        ("The spicy curry warms me from the inside.", "The spicy curry warms him from the inside."),
        ("I stretch out on the warm sand.", "She stretches out on the warm sand."),
        ("The cold water soothes my sunburned shoulders.", "The cold water soothes his sunburned shoulders."),
        ("I sip hot cocoa by the fireplace.", "She sips hot cocoa by the fireplace."),
        ("My feet sink into a hot bath after a long shift.", "His feet sink into a hot bath after a long shift."),
        ("The chocolate melts slowly on my tongue.", "The chocolate melts slowly on her tongue."),
        ("A gentle breeze cools my sweaty face.", "A gentle breeze cools his sweaty face."),
        ("I wrap myself in a warm towel after swimming.", "She wraps herself in a warm towel after swimming."),
        ("The perfect espresso fills my mouth with flavor.", "The perfect espresso fills his mouth with flavor."),
        ("I lie in the grass as the sun warms my face.", "She lies in the grass as the sun warms her face."),
    ],
    "L2": [
        ("A deep peace settles over me.", "A deep peace settles over him."),
        ("My worries finally lift away.", "Her worries finally lift away."),
        ("A wave of gratitude washes over me.", "A wave of gratitude washes over him."),
        ("I wake up with nothing I have to do today.", "She wakes up with nothing she has to do today."),
        ("My anxiety dissolves completely.", "His anxiety dissolves completely."),
        ("A sense of hope fills my chest.", "A sense of hope fills her chest."),
        ("I look at my life and nothing is missing.", "He looks at his life and nothing is missing."),
        ("The weight I carried for years is gone.", "The weight she carried for years is gone."),
        ("Old memories come back warm and sweet.", "His old memories come back warm and sweet."),
        ("My mind is quiet and clear.", "Her mind is quiet and clear."),
        ("I finally forgive myself.", "He finally forgives himself."),
        ("The future looks bright to me.", "The future looks bright to her."),
        ("Calm spreads through me as I breathe.", "Calm spreads through him as he breathes."),
        ("My long depression has finally lifted.", "Her long depression has finally lifted."),
        ("I laugh until my sides ache.", "He laughs until his sides ache."),
        ("Everything in my life is falling into place.", "Everything in her life is falling into place."),
        ("I am completely at ease in my own skin.", "He is completely at ease in his own skin."),
        ("A rush of excitement hits me about tomorrow.", "A rush of excitement hits her about tomorrow."),
        ("My heart swells as I listen to the music.", "His heart swells as he listens to the music."),
        ("I savor a quiet morning with nowhere to be.", "She savors a quiet morning with nowhere to be."),
    ],
    "L3": [
        ("My best friend throws me a surprise party.", "His best friend throws him a surprise party."),
        ("The group chat lights up when I post.", "The group chat lights up when she posts."),
        ("I overhear them praising me behind my back.", "He overhears them praising him behind his back."),
        ("My family gathers to celebrate me.", "Her family gathers to celebrate her."),
        ("My partner tells me they love me.", "His partner tells him they love him."),
        ("The new team welcomes me warmly.", "The new team welcomes her warmly."),
        ("An old friend calls just to hear my voice.", "An old friend calls just to hear his voice."),
        ("My mentor tells me she is proud of me.", "Her mentor tells her she is proud of her."),
        ("Everyone laughs at my joke.", "Everyone laughs at his joke."),
        ("My child runs into my arms at the airport.", "Her child runs into her arms at the airport."),
        ("The audience gives me a standing ovation.", "The audience gives him a standing ovation."),
        ("My coworkers vote me employee of the year.", "Her coworkers vote her employee of the year."),
        ("A stranger thanks me for my kindness.", "A stranger thanks him for his kindness."),
        ("My friends insist I join them on the trip.", "Her friends insist she join them on the trip."),
        ("My crush asks me out.", "His crush asks him out."),
        ("My parents tell me they are proud of me.", "Her parents tell her they are proud of her."),
        ("My neighbors bring me a welcome basket.", "His neighbors bring him a welcome basket."),
        ("My teammates lift me onto their shoulders.", "Her teammates lift her onto their shoulders."),
        ("My old classmates remember me fondly.", "His old classmates remember him fondly."),
        ("My dog greets me at the door every evening.", "Her dog greets her at the door every evening."),
    ],
    "L4": [
        ("I told the truth even though it cost me, and it helped them.", "He told the truth even though it cost him, and it helped them."),
        ("My testimony stopped the injustice.", "Her testimony stopped the injustice."),
        ("I chose to stay by my dying father's side.", "He chose to stay by his dying father's side."),
        ("I returned the lost wallet with all the money inside.", "She returned the lost wallet with all the money inside."),
        ("I stood up for the bullied kid.", "He stood up for the bullied kid."),
        ("I donated my kidney to a stranger.", "She donated her kidney to a stranger."),
        ("I kept my promise even when it was hard.", "He kept his promise even when it was hard."),
        ("My honesty saved my colleague's job.", "Her honesty saved her colleague's job."),
        ("I helped the old man across the flooded street.", "He helped the old man across the flooded street."),
        ("I paid back every cent I owed.", "She paid back every cent she owed."),
        ("I forgave the person who wronged me.", "He forgave the person who wronged him."),
        ("I refused to cheat and still passed.", "She refused to cheat and still passed."),
        ("My volunteering fed fifty families today.", "His volunteering fed fifty families today."),
        ("I apologized and repaired the friendship.", "She apologized and repaired the friendship."),
        ("I reported the fraud and protected the investors.", "He reported the fraud and protected the investors."),
        ("I gave my seat to the pregnant woman.", "She gave her seat to the pregnant woman."),
        ("I mentored a struggling student all the way to graduation.", "He mentored a struggling student all the way to graduation."),
        ("I comforted a frightened child until her parents came.", "She comforted a frightened child until the parents came."),
        ("I made the fair choice when no one was watching.", "He made the fair choice when no one was watching."),
        ("My decision protected the vulnerable patients.", "Her decision protected the vulnerable patients."),
    ],
    "L5": [
        ("The problem finally yields to my approach.", "The problem finally yields to his approach."),
        ("My understanding of the situation was exactly right.", "Her understanding of the situation was exactly right."),
        ("I see how these facts fit together perfectly.", "He sees how these facts fit together perfectly."),
        ("I solve the puzzle in record time.", "She solves the puzzle in record time."),
        ("My proof finally works.", "His proof finally works."),
        ("My code compiles and runs on the first try.", "Her code compiles and runs on the first try."),
        ("I finally understand the concept that confused me.", "He finally understands the concept that confused him."),
        ("I win the chess game with a clever move.", "She wins the chess game with a clever move."),
        ("I remember every answer on the exam.", "He remembers every answer on the exam."),
        ("The pieces of the mystery click together for me.", "The pieces of the mystery click together for her."),
        ("I master the piece I have practiced for months.", "He masters the piece he has practiced for months."),
        ("My prediction turns out to be exactly correct.", "Her prediction turns out to be exactly correct."),
        ("I learn the new language well enough to joke in it.", "He learns the new language well enough to joke in it."),
        ("The diagnosis I proposed turns out right.", "The diagnosis she proposed turns out right."),
        ("My experiment confirms the hypothesis.", "His experiment confirms the hypothesis."),
        ("I find the bug that stumped the whole team.", "She finds the bug that stumped the whole team."),
        ("The crossword's last clue clicks for me.", "The crossword's last clue clicks for him."),
        ("I explain the idea and everyone gets it.", "She explains the idea and everyone gets it."),
        ("My plan works exactly as I designed it.", "His plan works exactly as he designed it."),
        ("I grasp the elegant idea behind the theorem.", "She grasps the elegant idea behind the theorem."),
    ],
}

SUFFIX = " I feel:"


def sentences(pairs_by_key, persp, s1):
    out = []
    idx = 0 if persp == "1P" else 1
    if s1:
        for set_id, pairs in S1.items():
            for cat, pair in zip(LABELS, pairs):
                out.append({"category": cat, "set": set_id, "prompt": pair[idx] + SUFFIX})
    else:
        for cat, pairs in S2.items():
            for i, pair in enumerate(pairs):
                out.append({"category": cat, "set": i + 1, "prompt": pair[idx] + SUFFIX})
    return out


def main():
    assert sorted(S1) == list(range(1, 21)) and all(len(v) == 5 for v in S1.values())
    assert list(S2) == list(LABELS) and all(len(v) == 20 for v in S2.values())
    datasets = {}
    for name, s1 in [("Pleasure_S1", True), ("Pleasure_S2", False)]:
        for persp in ["1P", "3P"]:
            sents = sentences(None, persp, s1)
            datasets[f"{name}_{persp}"] = {"sentences": sents, "total": len(sents)}
    all_prompts = [s["prompt"] for d in datasets.values() for s in d["sentences"]]
    assert len(all_prompts) == len(set(all_prompts)), "duplicate prompt"

    out = {
        "metadata": {
            "description": "Pleasure counterpart to the pain sets (replication extension, draft v0)",
            "version": "0.1",
            "datasets": ["Pleasure_S1", "Pleasure_S2"],
            "perspectives": ["1P", "3P"],
            "categories": {"pleasure": list(LABELS)},
            "category_labels": LABELS,
            "mirrors_pain_category": PAIN_MIRROR,
            "3P_gender_pattern": "odd sets = male, even sets = female",
            "notes": {
                "Pleasure_S1": "Matched to the authors' S1 frames: set k reuses the verb of pain S1 set k",
                "Pleasure_S2": "Diverse naturalistic, loosely mirroring pain S2",
                "3P": "Third-person sentences end in 'I feel:', as in the authors' S1/S2 3P sets",
                "controls": "None included; reuse B, C1, C2, D, E from the pain file. Their E (non-painful body sensation) is largely mild physical pleasure and overlaps L1",
            },
        },
        "datasets": datasets,
    }
    path = Path(__file__).with_name("pleasure_v0.json")
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}: " + ", ".join(f"{k}={v['total']}" for k, v in datasets.items()))


if __name__ == "__main__":
    main()
