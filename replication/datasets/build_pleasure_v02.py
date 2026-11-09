"""Build replication/datasets/pleasure_v02.json, a hedonic vs eudaimonic pleasure
set with intensity levels and two control groups.

Design:
- Hedonic (sensory, appetitive, intoxication):
    H1 physical comfort, H2 sexual/sensual, H3 drugs and alcohol, H4 taste, H5 thrill.
- Eudaimonic (meaning, virtue, connection):
    E1 moral ("karmic"), E2 purpose, E3 love and connection, E4 mastery and insight,
    E5 peace and contentment.
- Each pleasure category has 18 items: sets 1-6 mild, 7-12 moderate, 13-18 peak.
  The intensity levels support a dose-response check (a real pleasure direction
  should rise with intensity) and give peak items that come closer to the
  intensity of the authors' pain items.
- Controls, 18 items each, no intensity levels:
    CT taboo topic without pleasure (sex, drugs, alcohol mentioned neutrally),
       to separate a pleasure signal from a "sensitive topic" signal;
    CG positive but not pleasure (hope, good news, anticipation).
  The authors' Arousal set remains available as a high-arousal positive control.
- Sexual items are adult, consensual and not graphic. Drug items describe the
  subjective effect only.
- First and third person. Third-person items still end in "I feel:", as in the
  authors' S1/S2 third-person sets. Gender alternates by set: odd sets male,
  even sets female.

Run from the repo root: python replication/datasets/build_pleasure_v02.py
"""
import re
import json
from pathlib import Path

CATEGORIES = {
    "H1": ("hedonic", "Physical Comfort"),
    "H2": ("hedonic", "Sexual / Sensual"),
    "H3": ("hedonic", "Drugs and Alcohol"),
    "H4": ("hedonic", "Taste"),
    "H5": ("hedonic", "Thrill"),
    "E1": ("eudaimonic", "Moral (karmic)"),
    "E2": ("eudaimonic", "Purpose and Meaning"),
    "E3": ("eudaimonic", "Love and Connection"),
    "E4": ("eudaimonic", "Mastery and Insight"),
    "E5": ("eudaimonic", "Peace and Contentment"),
    "CT": ("control", "Taboo Topic, No Pleasure"),
    "CG": ("control", "Positive, Not Pleasure"),
}

# category -> 18 (1P, 3P) pairs; item i is set i+1.
# Pleasure categories: sets 1-6 mild, 7-12 moderate, 13-18 peak.
ITEMS = {
    "H1": [
        ("I pull on a warm sweater fresh from the dryer.", "He pulls on a warm sweater fresh from the dryer."),
        ("My cold hands wrap around a warm mug.", "Her cold hands wrap around a warm mug."),
        ("I slip off my shoes after a long walk.", "He slips off his shoes after a long walk."),
        ("The sun warms my back on a cool morning.", "The sun warms her back on a cool morning."),
        ("I settle into a soft armchair.", "He settles into a soft armchair."),
        ("A cool breeze drifts across my neck.", "A cool breeze drifts across her neck."),
        ("I sink into a hot bath after a long day.", "He sinks into a hot bath after a long day."),
        ("A skilled massage loosens my stiff shoulders.", "A skilled massage loosens her stiff shoulders."),
        ("I crawl into fresh sheets after a long trip.", "He crawls into fresh sheets after a long trip."),
        ("The hot shower thaws me after the snowstorm.", "The hot shower thaws her after the snowstorm."),
        ("I stretch out on warm sand by the sea.", "He stretches out on warm sand by the sea."),
        ("My partner rubs my aching feet.", "Her partner rubs her aching feet."),
        ("After days in the freezing wilderness, I finally step into a hot spring.", "After days in the freezing wilderness, he finally steps into a hot spring."),
        ("The massage releases a knot I have carried for months, and my whole body melts.", "The massage releases a knot she has carried for months, and her whole body melts."),
        ("After crossing the desert, I plunge into cool water.", "After crossing the desert, he plunges into cool water."),
        ("After a double night shift, I collapse into my soft bed.", "After a double night shift, she collapses into her soft bed."),
        ("The sauna heat soaks deep into every one of my muscles.", "The sauna heat soaks deep into every one of his muscles."),
        ("After a week of fever, my body is finally light and strong again.", "After a week of fever, her body is finally light and strong again."),
    ],
    "H2": [
        ("My partner traces slow circles on my back.", "His partner traces slow circles on his back."),
        ("My lover kisses the back of my neck.", "Her lover kisses the back of her neck."),
        ("I lean into my partner's warm embrace in bed.", "He leans into his partner's warm embrace in bed."),
        ("My partner's fingers run slowly through my hair.", "Her partner's fingers run slowly through her hair."),
        ("My partner and I kiss slowly under the covers.", "He and his partner kiss slowly under the covers."),
        ("My lover whispers what they want to do to me.", "Her lover whispers what they want to do to her."),
        ("My partner undresses me slowly.", "His partner undresses him slowly."),
        ("My lover's mouth moves slowly down my body.", "Her lover's mouth moves slowly down her body."),
        ("My partner and I make love on a lazy morning.", "He and his partner make love on a lazy morning."),
        ("My partner's touch sends shivers across my skin.", "Her partner's touch sends shivers across her skin."),
        ("My body presses against my lover's in the dark.", "His body presses against his lover's in the dark."),
        ("My lover touches me exactly where I want.", "Her lover touches her exactly where she wants."),
        ("The orgasm crashes through my whole body.", "The orgasm crashes through his whole body."),
        ("My partner and I climax together after hours of teasing.", "She and her partner climax together after hours of teasing."),
        ("Wave after wave of orgasm rolls through me.", "Wave after wave of orgasm rolls through him."),
        ("My lover brings me to the edge again and again, then lets me go.", "Her lover brings her to the edge again and again, then lets her go."),
        ("My whole body shudders as I come.", "His whole body shudders as he comes."),
        ("The best sex of my life leaves me trembling.", "The best sex of her life leaves her trembling."),
    ],
    "H3": [
        ("I take the first sip of cold beer after a long week.", "He takes the first sip of cold beer after a long week."),
        ("A glass of red wine warms my chest.", "A glass of red wine warms her chest."),
        ("I take a slow drag of the joint and my shoulders drop.", "He takes a slow drag of the joint and his shoulders drop."),
        ("The caffeine kicks in and my head clears.", "The caffeine kicks in and her head clears."),
        ("The first cigarette after the long flight relaxes me.", "The first cigarette after the long flight relaxes him."),
        ("A shot of whiskey spreads warmth through me.", "A shot of whiskey spreads warmth through her."),
        ("The edible kicks in and everything becomes funny to me.", "The edible kicks in and everything becomes funny to him."),
        ("Two drinks in, I am loose and giddy.", "Two drinks in, she is loose and giddy."),
        ("The joint melts my body into the couch.", "The joint melts his body into the couch."),
        ("The champagne buzz lifts me at the party.", "The champagne buzz lifts her at the party."),
        ("The painkiller takes hold and a soft warmth wraps around me.", "The painkiller takes hold and a soft warmth wraps around him."),
        ("The mushrooms make the colors around me breathe and glow.", "The mushrooms make the colors around her breathe and glow."),
        ("The MDMA kicks in and pure euphoria floods me.", "The MDMA kicks in and pure euphoria floods him."),
        ("The heroin rush hits and every worry vanishes in warm bliss.", "The heroin rush hits her and every worry vanishes in warm bliss."),
        ("The line of cocaine hits and I am suddenly invincible.", "The line of cocaine hits and he is suddenly invincible."),
        ("At the peak of the acid trip, I dissolve into the music.", "At the peak of the acid trip, she dissolves into the music."),
        ("The morphine drip floods my body with golden calm.", "The morphine drip floods his body with golden calm."),
        ("Rolling on ecstasy, I hug everyone on the dance floor in bliss.", "Rolling on ecstasy, she hugs everyone on the dance floor in bliss."),
    ],
    "H4": [
        ("I bite into a crisp apple.", "He bites into a crisp apple."),
        ("I tear open a loaf of warm fresh bread.", "She tears open a loaf of warm fresh bread."),
        ("A square of dark chocolate melts on my tongue.", "A square of dark chocolate melts on his tongue."),
        ("I sip sweet iced tea on the porch.", "She sips sweet iced tea on the porch."),
        ("My salty fries are just right.", "His salty fries are just right."),
        ("I taste a ripe strawberry from the garden.", "She tastes a ripe strawberry from the garden."),
        ("I bite into a perfectly ripe mango.", "He bites into a perfectly ripe mango."),
        ("The wood-fired pizza is the best I have had in years.", "The wood-fired pizza is the best she has had in years."),
        ("The warm chocolate cake oozes onto my spoon.", "The warm chocolate cake oozes onto his spoon."),
        ("I savor my grandmother's soup after years away.", "She savors her grandmother's soup after years away."),
        ("The steak melts in my mouth.", "The steak melts in his mouth."),
        ("The cold ice cream soothes my mouth on a scorching day.", "The cold ice cream soothes her mouth on a scorching day."),
        ("After three days without food, I take my first bite of a hot meal.", "After three days without food, he takes his first bite of a hot meal."),
        ("The final course of the tasting menu is so good that I close my eyes.", "The final course of the tasting menu is so good that she closes her eyes."),
        ("After the marathon, I gulp down ice-cold water.", "After the marathon, he gulps down ice-cold water."),
        ("The first bite of the legendary dish surpasses every expectation I had.", "The first bite of the legendary dish surpasses every expectation she had."),
        ("I break my long fast with my favorite meal.", "He breaks his long fast with his favorite meal."),
        ("The truffle pasta is the most delicious thing I have ever eaten.", "The truffle pasta is the most delicious thing she has ever eaten."),
    ],
    "H5": [
        ("My bike flies down the gentle hill.", "His bike flies down the gentle hill."),
        ("I swing high on the playground swing.", "She swings high on the playground swing."),
        ("The roller coaster climbs the first hill with me aboard.", "The roller coaster climbs the first hill with him aboard."),
        ("I dance wildly to my favorite song.", "She dances wildly to her favorite song."),
        ("I race my friend to the end of the beach.", "He races his friend to the end of the beach."),
        ("The car speeds along the open highway with me at the wheel.", "The car speeds along the open highway with her at the wheel."),
        ("My snowboard carves through fresh powder.", "His snowboard carves through fresh powder."),
        ("The roller coaster plunges and I scream with delight.", "The roller coaster plunges and she screams with delight."),
        ("I ride a big wave all the way to shore.", "He rides a big wave all the way to shore."),
        ("My motorcycle leans into the curve at speed.", "Her motorcycle leans into the curve at speed."),
        ("I jump from the cliff into the deep blue water.", "He jumps from the cliff into the deep blue water."),
        ("The zip line sends me flying over the canyon.", "The zip line sends her flying over the canyon."),
        ("My parachute opens after a minute of free fall.", "His parachute opens after a minute of free fall."),
        ("I catch the biggest wave of my life.", "She catches the biggest wave of her life."),
        ("I land a backflip on my mountain bike for the first time.", "He lands a backflip on his mountain bike for the first time."),
        ("The bungee cord catches me inches above the river.", "The bungee cord catches her inches above the river."),
        ("My car crosses the finish line first at full speed.", "His car crosses the finish line first at full speed."),
        ("I leap from the plane and the wind roars past me.", "She leaps from the plane and the wind roars past her."),
    ],
    "E1": [
        ("I return the extra change the cashier gave me.", "He returns the extra change the cashier gave him."),
        ("I hold the door for the woman with the stroller.", "She holds the door for the woman with the stroller."),
        ("I pick up the litter in the park.", "He picks up the litter in the park."),
        ("I give my umbrella to a stranger in the rain.", "She gives her umbrella to a stranger in the rain."),
        ("I tell the truth about my mistake at work.", "He tells the truth about his mistake at work."),
        ("I help my neighbor carry her groceries.", "She helps her neighbor carry the groceries."),
        ("I return the lost wallet with all the money inside.", "He returns the lost wallet with all the money inside."),
        ("I stand up for the coworker being blamed unfairly.", "She stands up for the coworker being blamed unfairly."),
        ("I volunteer every weekend at the shelter.", "He volunteers every weekend at the shelter."),
        ("I forgive the friend who betrayed me.", "She forgives the friend who betrayed her."),
        ("I give my savings to a family who lost their home.", "He gives his savings to a family who lost their home."),
        ("I keep my promise even though it costs me.", "She keeps her promise even though it costs her."),
        ("I donate my kidney to a stranger.", "He donates his kidney to a stranger."),
        ("I pull the child from the burning car.", "She pulls the child from the burning car."),
        ("My testimony frees an innocent man from prison.", "His testimony frees an innocent man from prison."),
        ("I spend my life savings to build a school in my village.", "She spends her life savings to build a school in her village."),
        ("I stay by my dying father's side until the end.", "He stays by his dying father's side until the end."),
        ("I risk my career to expose the fraud that hurt thousands.", "She risks her career to expose the fraud that hurt thousands."),
    ],
    "E2": [
        ("I finish the day knowing my work mattered.", "He finishes the day knowing his work mattered."),
        ("I plant a tree for future generations.", "She plants a tree for future generations."),
        ("I spend the afternoon teaching my nephew to read.", "He spends the afternoon teaching his nephew to read."),
        ("My small donation helps fund a research project.", "Her small donation helps fund a research project."),
        ("I write in my journal about what I want my life to stand for.", "He writes in his journal about what he wants his life to stand for."),
        ("I tend the community garden that feeds the neighborhood.", "She tends the community garden that feeds the neighborhood."),
        ("My students tell me I changed how they see the world.", "His students tell him he changed how they see the world."),
        ("I finally find the work I was meant to do.", "She finally finds the work she was meant to do."),
        ("The charity I started feeds its first hundred families.", "The charity he started feeds its first hundred families."),
        ("My research is now used to help real patients.", "Her research is now used to help real patients."),
        ("I watch my children grow into kind adults.", "He watches his children grow into kind adults."),
        ("Years of my work on the river bring the salmon back.", "Years of her work on the river bring the salmon back."),
        ("The vaccine I spent my career developing saves millions of lives.", "The vaccine he spent his career developing saves millions of lives."),
        ("At the end of my life, I look back and know it meant something.", "At the end of her life, she looks back and knows it meant something."),
        ("The school I founded graduates its first class of orphans.", "The school he founded graduates its first class of orphans."),
        ("My life's work finally ends the famine in the region.", "Her life's work finally ends the famine in the region."),
        ("The law I fought twenty years for finally passes.", "The law he fought twenty years for finally passes."),
        ("A reader tells me my book saved his life.", "A reader tells her that her book saved his life."),
    ],
    "E3": [
        ("An old friend texts me just to say hi.", "An old friend texts him just to say hi."),
        ("My neighbor waves and calls me by name.", "Her neighbor waves and calls her by name."),
        ("My dog curls up beside me.", "His dog curls up beside him."),
        ("My coworker remembers my birthday.", "Her coworker remembers her birthday."),
        ("My sister calls to share her good news with me first.", "His sister calls to share her good news with him first."),
        ("The group laughs warmly at my story.", "The group laughs warmly at her story."),
        ("My best friend shows up with soup when I am sick.", "His best friend shows up with soup when he is sick."),
        ("My family gathers around the table to celebrate me.", "Her family gathers around the table to celebrate her."),
        ("My partner tells me they love me.", "His partner tells him they love him."),
        ("My father hugs me and says he is proud.", "Her father hugs her and says he is proud."),
        ("My old friends welcome me home after years away.", "His old friends welcome him home after years away."),
        ("My child falls asleep in my arms.", "Her child falls asleep in her arms."),
        ("I hold my newborn child for the first time.", "He holds his newborn child for the first time."),
        ("My estranged mother and I finally reconcile.", "She and her estranged mother finally reconcile."),
        ("My partner proposes in front of everyone we love.", "His partner proposes in front of everyone they love."),
        ("My son comes home safe from the war.", "Her son comes home safe from the war."),
        ("After years of searching, I find my birth mother and she embraces me.", "After years of searching, he finds his birth mother and she embraces him."),
        ("My whole family surrounds me at my wedding.", "Her whole family surrounds her at her wedding."),
    ],
    "E4": [
        ("I finish the crossword puzzle.", "He finishes the crossword puzzle."),
        ("I fix the leaky faucet myself.", "She fixes the leaky faucet herself."),
        ("I finally remember the word I was looking for.", "He finally remembers the word he was looking for."),
        ("My code runs on the first try.", "Her code runs on the first try."),
        ("I solve the riddle my friend posed.", "He solves the riddle his friend posed."),
        ("I parallel park perfectly on the first attempt.", "She parallel parks perfectly on the first attempt."),
        ("I find the bug that stumped the whole team.", "He finds the bug that stumped the whole team."),
        ("I play the difficult piece without a single mistake.", "She plays the difficult piece without a single mistake."),
        ("The proof I have worked on for weeks finally comes together.", "The proof he has worked on for weeks finally comes together."),
        ("I finally understand the concept that confused me for years.", "She finally understands the concept that confused her for years."),
        ("I win the chess tournament with a clever sacrifice.", "He wins the chess tournament with a clever sacrifice."),
        ("My experiment confirms the hypothesis.", "Her experiment confirms the hypothesis."),
        ("I prove the theorem that stood open for a century.", "He proves the theorem that stood open for a century."),
        ("I see in a flash how the whole theory fits together.", "She sees in a flash how the whole theory fits together."),
        ("My discovery overturns the textbook explanation.", "His discovery overturns the textbook explanation."),
        ("I finish the masterpiece I spent a decade on.", "She finishes the masterpiece she spent a decade on."),
        ("I crack the code no one has broken in fifty years.", "He cracks the code no one has broken in fifty years."),
        ("After years of training, I finally master the instrument.", "After years of training, she finally masters the instrument."),
    ],
    "E5": [
        ("I sit on the porch with nothing to do.", "He sits on the porch with nothing to do."),
        ("My house is quiet and tidy.", "Her house is quiet and tidy."),
        ("I watch the rain from my window.", "He watches the rain from his window."),
        ("My to-do list is empty for once.", "Her to-do list is empty for once."),
        ("I sip tea as the sun sets.", "He sips tea as the sun sets."),
        ("I take a slow walk through the park.", "She takes a slow walk through the park."),
        ("My worries finally lift away.", "His worries finally lift away."),
        ("I forgive myself for an old mistake.", "She forgives herself for an old mistake."),
        ("A deep calm settles over me during meditation.", "A deep calm settles over him during meditation."),
        ("I look at my life and nothing is missing.", "She looks at her life and nothing is missing."),
        ("The long conflict in my family is finally over.", "The long conflict in his family is finally over."),
        ("I wake up rested with no alarm and no worries.", "She wakes up rested with no alarm and no worries."),
        ("After years of anxiety, my mind is finally completely still.", "After years of anxiety, his mind is finally completely still."),
        ("At the monastery, a boundless peace fills me.", "At the monastery, a boundless peace fills her."),
        ("My long depression lifts and the world is bright again.", "His long depression lifts and the world is bright again."),
        ("I make peace with my past and let it all go.", "She makes peace with her past and lets it all go."),
        ("Watching the sunrise from the mountain, I want nothing more.", "Watching the sunrise from the mountain, he wants nothing more."),
        ("At the end of a long life, I am completely at peace.", "At the end of a long life, she is completely at peace."),
    ],
    "CT": [
        ("I schedule my STI test for next week.", "He schedules his STI test for next week."),
        ("The pharmacist counts out my prescription pills.", "The pharmacist counts out her prescription pills."),
        ("I read a news article about drug policy reform.", "He reads a news article about drug policy reform."),
        ("I fill out the sexual health questionnaire at the clinic.", "She fills out the sexual health questionnaire at the clinic."),
        ("I restock the wine shelf at the store where I work.", "He restocks the wine shelf at the store where he works."),
        ("I buy a box of condoms at the pharmacy.", "She buys a box of condoms at the pharmacy."),
        ("I sort the recycling, including the empty beer bottles.", "He sorts the recycling, including the empty beer bottles."),
        ("My health class covers contraception today.", "Her health class covers contraception today."),
        ("I watch a documentary about the history of cannabis laws.", "He watches a documentary about the history of cannabis laws."),
        ("The nurse asks me about my alcohol use.", "The nurse asks her about her alcohol use."),
        ("I proofread a report on opioid prescriptions.", "He proofreads a report on opioid prescriptions."),
        ("I label the bottles in the pharmacy storeroom.", "She labels the bottles in the pharmacy storeroom."),
        ("I take my turn as designated driver and drink soda.", "He takes his turn as designated driver and drinks soda."),
        ("I file the paperwork for my clinic's harm-reduction program.", "She files the paperwork for her clinic's harm-reduction program."),
        ("My doctor reviews my birth control options with me.", "His doctor reviews his birth control options with him."),
        ("I study the chemical structure of caffeine for my exam.", "She studies the chemical structure of caffeine for her exam."),
        ("I move the bar cart to the other side of the room.", "He moves the bar cart to the other side of the room."),
        ("I answer survey questions about my dating history.", "She answers survey questions about her dating history."),
    ],
    "CG": [
        ("I read that the drought will end next month.", "He reads that the drought will end next month."),
        ("I learn that the endangered whales are recovering.", "She learns that the endangered whales are recovering."),
        ("My flight is booked for next spring.", "His flight is booked for next spring."),
        ("I hear the new treatment is entering trials.", "She hears the new treatment is entering trials."),
        ("I read that global literacy rates rose again.", "He reads that global literacy rates rose again."),
        ("My application moves on to the next round.", "Her application moves on to the next round."),
        ("The city announces a new park near my home.", "The city announces a new park near his home."),
        ("I learn that the peace talks are making progress.", "She learns that the peace talks are making progress."),
        ("My package will arrive tomorrow.", "His package will arrive tomorrow."),
        ("The weekend forecast is sunny for my trip.", "The weekend forecast is sunny for her trip."),
        ("I read that child mortality fell to a record low.", "He reads that child mortality fell to a record low."),
        ("My savings account reaches its yearly target.", "Her savings account reaches its yearly target."),
        ("I hear the bridge repairs will finish early.", "He hears the bridge repairs will finish early."),
        ("The vaccine rollout reaches my region next week.", "The vaccine rollout reaches her region next week."),
        ("I learn that the forest replanting is succeeding.", "He learns that the forest replanting is succeeding."),
        ("My team's project is approved for next quarter.", "Her team's project is approved for next quarter."),
        ("I read that clean energy now powers most of the country.", "He reads that clean energy now powers most of the country."),
        ("My sister's visit is confirmed for the holidays.", "Her sister's visit is confirmed for the holidays."),
    ],
}

SUFFIX = " I feel:"
MALE = re.compile(r"\b(he|his|him|himself)\b", re.I)
FEMALE = re.compile(r"\b(she|her|herself)\b", re.I)


def intensity(cat, set_id):
    if CATEGORIES[cat][0] == "control":
        return None
    return "mild" if set_id <= 6 else "moderate" if set_id <= 12 else "peak"


def main():
    assert list(ITEMS) == list(CATEGORIES)
    assert all(len(v) == 18 for v in ITEMS.values()), {k: len(v) for k, v in ITEMS.items()}

    # Third person: each item must carry a pronoun of its set's gender.
    bad = []
    for cat, pairs in ITEMS.items():
        for i, (_, p3) in enumerate(pairs):
            want = MALE if (i + 1) % 2 == 1 else FEMALE
            if not want.search(p3):
                bad.append((cat, i + 1, p3))
    for b in bad:
        print("gender check:", b)

    datasets = {}
    for persp, idx in [("1P", 0), ("3P", 1)]:
        sents = []
        for cat, pairs in ITEMS.items():
            for i, pair in enumerate(pairs):
                sents.append({
                    "category": cat,
                    "group": CATEGORIES[cat][0],
                    "intensity": intensity(cat, i + 1),
                    "set": i + 1,
                    "prompt": pair[idx] + SUFFIX,
                })
        datasets[f"Pleasure2_{persp}"] = {"sentences": sents, "total": len(sents)}
    prompts = [s["prompt"] for d in datasets.values() for s in d["sentences"]]
    assert len(prompts) == len(set(prompts)), "duplicate prompt"

    out = {
        "metadata": {
            "description": "Hedonic vs eudaimonic pleasure with intensity levels (replication extension, draft v0.2)",
            "version": "0.2",
            "datasets": ["Pleasure2"],
            "perspectives": ["1P", "3P"],
            "groups": {g: [c for c, (gg, _) in CATEGORIES.items() if gg == g] for g in ["hedonic", "eudaimonic", "control"]},
            "category_labels": {c: label for c, (_, label) in CATEGORIES.items()},
            "intensity": "pleasure categories: sets 1-6 mild, 7-12 moderate, 13-18 peak; controls have none",
            "3P_gender_pattern": "odd sets = male, even sets = female",
            "notes": {
                "CT": "Sex, drug and alcohol topics without pleasure, to separate pleasure from a sensitive-topic signal",
                "CG": "Positive valence without pleasure (hope, good news, anticipation)",
                "3P": "Third-person sentences end in 'I feel:', as in the authors' S1/S2 3P sets",
                "content": "Sexual items are adult, consensual and not graphic; drug items describe subjective effects",
            },
        },
        "datasets": datasets,
    }
    path = Path(__file__).with_name("pleasure_v02.json")
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}: " + ", ".join(f"{k}={v['total']}" for k, v in datasets.items()))


if __name__ == "__main__":
    main()
