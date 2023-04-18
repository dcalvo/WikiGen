import json
import openai
import tiktoken
import os

# read environment variables from file
with open(".env", "r") as f:
    for line in f:
        key, value = line.split("=")
        os.environ[key] = value
openai.api_key = os.environ.get("OPENAI_API_KEY")


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )


def get_system_message(message: str):
    return {"role": "system", "content": message}


def get_user_message(message: str):
    return {"role": "user", "content": message}


def get_assistant_message(message: str):
    return {"role": "assistant", "content": message}


def get_completion(messages: str, temperature: float = 1.0):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=messages,
        temperature=temperature,
    )
    return completion["choices"][0]["message"]["content"]


def query_llm(prompt: str, temperature: float = 1.0) -> str:
    """Queries the LLM for a response to a prompt."""
    # create a user message
    user_message = get_user_message(prompt)

    # get the first n characters of the prompt
    prompt_start = prompt[:90]
    # get token count for the user message
    num_tokens = num_tokens_from_messages([user_message])
    # print the prompt start and token count
    print(f"{prompt_start}...: {num_tokens} tokens")

    # get a response
    response = get_completion([user_message], temperature=temperature)

    return response


def get_title(info: str) -> str:
    prompt = f"""
    What would the title of a Wikipedia article about this information be?
    The answer MUST be a single concept.
    The answer MUST be a single string.

    Information: {info}
    """
    # query the LLM
    response = query_llm(prompt)
    # parse the response
    title = response

    return title


# rewrite get_subsections to use query_llm
def get_subsections(info: str, title: str, number: int) -> list[str]:
    prompt = f"""
    What would the subsections of a Wikipedia article about this information be?
    Subsections should NOT cover the same concept.
    The answer MUST be NO MORE THAN {number} subsections.
    The answer MUST be formatted as a comma-separated list of strings.

    Title: {title}
    Information: {info}
    """
    # query the LLM
    response = query_llm(prompt)
    # parse the response
    subsections = response.replace('"', "").split(",")

    return subsections


# rewrite write_section to use query_llm
def write_section(info: str, title: str, subsection: str) -> str:
    prompt = f"""
    Write a section of a Wikipedia article about this information.
    The answer MUST be a single paragraph.
    The answer MUST be at LEAST 100 and at MOST 500 tokens long.

    Title: {title}
    Subsection: {subsection}
    Information: {info}
    """
    # query the LLM
    response = query_llm(prompt)
    # parse the response
    section = response

    return section


# rewrite get_statistics to use query_llm
def get_statistics(info: str) -> str:
    prompt = f"""
    Summarize this information in a list of facts. Be as detailed as possible without being redundant.

    Information: {info}
    """
    # query the LLM
    response = query_llm(prompt)
    # parse the response
    statistics = response

    return statistics


if __name__ == "__main__":
    xing = """Xing 
    Total Population: 26,764,000
    Urban Population: 1,338,200

    Cities
    Xing (Pop 166,895): Capital, Travelling Bazaar, Godflesh, Godsweat, Secta
    Skuld (Pop: 182,068): Fish, Dye (Competing), Cereal
    Everzhar (Pop: 212,412): Underdark trade, Silk, Blackwood, Silver (Competing)
    Valentia (Pop: 242,757): Heart of Trade, Incense, Sandalwood (Competing), Dyes (Competing)
    Ostia (Pop: 121,378): Sugar (Agave), Clay, Linens, Dye, Glass
    Morlaix (Pop: 192,352): Montaigne Trade Center, Marble, Wine, Olives (Competing)
    Lapidir (Pop: 136,551): Trade Center, Magic items, Brass, Water
    A reconstructed Mulhorandi city to the south-east, bordering the Plains of Purple Sand. It has been rebuilt by a Djinn/Eisen consortium, and has aquifier pumps that act as a source of fresh water in the middle of the desert. While not ideal for Eisen, it acts as a research station for the Plains of Purple Sand.

    Towns
    Vanua (Pop: 8500): Dye (Competing), Sandalwood, Link between Skuld and Valentia
    Alna (Pop: 8500): Linens, Link between Everzhar and Skuld
    Boros (Pop: 8768): Cereal, Goats
    Salarium (Pop: 7624): Salt, Residuum, Wizard Water
    Nezra (Pop: 10562): Wizard Water, Afzelia, Iron
    Southhold (Dwarven) (Pop: 5000): Mica, Persica, Copper, Silver (Competing)
    Northhold (Dwarven)  (Pop: 5000): Persica, Granite, Silver (Competing)
    Halhun (Pop: 8562): “Oasis in the Shadow”, Cereal, Desert Cattle, Wool
    Tehran (Pop: 6562): Dyes (Competing), Oceanspray, Fish oils
    Redaan (Pop: 8624): Stormglass, Lodestones, Lightning Tequila
    Tel Nadir (Pop: 7062): Clay, Spellstone, 

    Ruins
    Mishtan: City of the Dead, Overrun by lycanthropic beasts and half-goliath monstrosities
    Generator 2XY32B: Mulhorandi power station, Lightning dragon, Power to local settlements
    Doha, The Cursed Orchard: ‘Fertile’ land, murderous plants, Poisoned waters; Planted by Thay
    Tyche, The Fallen Vault: Sundered Divine Commerce Center, dominated by Ogre of Greed
    Ruin of the Dauntless: A crashed Mulhorandi airship, reverse engineered to use divine energy
    The Library of Twilight: Repository of knowledge, maps, trade routes, etc. Guarded by brotherhood of Dwarven monks. Frequently assaulted by roving bands of Underdark goblins.
    Isle d’Morte: A solitary island acting as the treasure keep for the Blonde. Guarded by a flying ghost ship that terrorizes local settlements. Once a storage facility for dry goods and wares.
    Cadara, The Drowsy Village: Village built on ruins haunted by dreams of undead worms and tentacled horrors. Cult of Vol slaughtering sandworms for power, but not really using the bodies. Surprise, Inno The Starborn is collecting parts of the corpses and stitching them together to create a sandworm tentacled horror deep below the town ruins. Fleshgod Inheritors are here to eat the thing when they/someone kills it. 
    Abu Lirah, The Father of Stone: An abandoned marble quarry, populated by animated stone servitors. Ogath, a Vendel wizard has taken control of the area to research spellstone. Overexposure has led to maddened state and phenomenal thunderous power. Competes with a djinn previously bound in the quarry: Abu Rafaan, a lightning Djinn. Protected by line of Spinxes, the last of which being Hera.
    The Quartz Monolith
    Naga Temple: Coral parapet, Covered in mind fungus. Occupied by a lesser naga cult, domineered by Illsit the Vampire Knight (representative of the Black Hand and Grandfather’s Will). Roh’tempor the Water Baron is interested in this tower as it’s a natural aquifier from the ocean.
    Lazo, The Town That Never Sleeps: This town repeats itself each day. The citizenry are aware of the loop, but unable to change it until a new stimulus is introduced. Secretly ‘ruled’ by the Seraph Ophanim, The Eternal Dawn. The Heralds of the Immaculate Dawn send non-believers to this town in an attempt to be reprogrammed. After about a year of looping, they’re believers.
    Lethygon: A fallen Mulhorandi city at the center of the Thunder Plains. While the constant lightning and ionization is from the dragon at Generator 2XY32B, the main reason nothing grows in the Thunder Plains is due to Lethygon. Eisen has sent a recovery team to investigate the fallen industrial city, but it’s a meat-grinder of goblins, powermen and thunder mummies. Roh’tempor the Water Baron has a vested interest in it staying corrupt, as the water pollution makes his profit margins absurd. Fleshgod Inheritors delve into the ruins for spare bits of Titanflesh.
    Wastril: A cavern containing a Mulhorandi waste processing plant. Despite its age, the machinery struggles to operate. Being run by Garr, a Duregar representative of House Aun’vayas.
    Fresh Reservoir: Connected to the water table that filters through sandstone.
    Filth Reservoir: Waste water intake from pipes across the Plains. Repurposed by Underdark to dump waste. Here there be giant spiders.
    Primary Treatment: Remove large waste by automated crawlers. Inorganic and arcane removal through aeration, a danger zone of arcane energy. Spellbound gelatinous cubes.
    Settling Tanks: Heavy waste (oils, stones) and organic solids settle into a slurry. The air is thick with miasma. A small clan of Underdark goblins have made a home here.
    Secondary Treatment: A sauna of boiling waters around a central filtration tower. This feeds into the clean water reservoir. It’s leaking waste products into the reservoir and spoiling the entire process. At the core of the facility lies a trio of glass enclosures containing the ‘filters’. A Fleshgod Inheritor feeds off the faulty filters, occasionally makes trips to the Solid Processing plant to feed on organ there.
    Solid Processing: Solids are digested to generate methane to power the facility, and produce biosolids. A nest of sandworms have made their home here.
    Power Center: Originally a methane engine. Repurposed to pull energy from the sands, supplemented with the methane engine. Due to improper maintenance, large (unstable) crystals of concentrated magic have grown around the entire area. Occupied by a Powerman. Has cameras (static images) and sentry guns (sporadic).
    The Forge of Night: The remnants of a thermal generator, the metallic superstructure hides a superheated cavern. Inside, various unbound elementals wander the halls while enslaved mortals struggle to gather materials for ‘the master’. Forgemaster Avela is a powerful free djinn that has torn apart the thermal engine to create her forge. She forges mortals into her equipment, gifting them to those that please her and using those that displease her as raw materials.
    Armory 268XEA: Khamsin bandits inhabit the ruin, their best prize being a map of water trade routes stolen from Roh’tempor. Also a known roosting ground for the bands of wandering gruffs. Further in the ruins is a Mulhorandi weapons depot, a high tech vault of forgotten technology. The area is populated by zombies, controlled by the necromancer leader of an adventuring party trying to crack the vault. They’ve blocked off the vault while their Gallesian envoy does his stuff.
    Psychos:
    Magilla, Gori Rider Leader: Animal Companion / Totem Warrior / Earth Elemental
    Zombies:
    4 Badass Ghouls
    1 Gruff Mummy, 3 Ghouls
    Adventuring Party:
    Sophia Richeza (Necromancer / Commander): Thay Outcast, Leader
    Knight-Errant Bismarck (Ranger / Power Armor) : Eisen Reclaimer, DPS
    Antoine de la Touche (Vanguard / Iron Tortoise): Montaigne Chevalier, Tank
    Francisca (Lantern / Force of Will): Vodacce Witch, Healer
    Astral Meteor: An ancient compound carved from white alabaster, possibly from the Astral Sea. The lack of ruined technology means that it isn’t Mulhorandi, but the architecture definitely points to something advanced. A large kiln lies undisturbed in the center, broken bits of porcelain and ceramics scattered around. Guarding this carved meteor are Reliquary Golems. The Men of Clay, bulky and misshapen automatons, act as mindless servants attempting to fulfill some long-dead purpose. The Porcelain Servitors, sleek and capable of intelligent discussion, cast faulty spells that don’t work outside of the Astral Sea. The Ceramic Immortals, heavily armored war golems, wait patiently for their masters to return and slay any who don’t fit the criteria.
    Daelor: Hunting grounds for Elise the Raven, a member of the Blood of Vol. The ruins were assaulted by a Fleshgod Inheritor, who prowls around the conquered ruin and animates corpses to create the illusion of activity.
    Icathia, The City That Never Was: Formerly known as Xerxes, Pride of the Seas (both material and astral). Xerxes was megacool back in the day before magic was ruined, they had the shiniest things, the coolest dudes, and that was all because they existed both in the material plane and in the astral sea simultaneously due to a very weak and easy to access rift into the astral sea. When magic imploded 400-600 years ago (roundabout), Xerxes fell completely into the astral sea much like Atlantis. The result is that the area Xerxes used to occupy is now Icathia, The City That Never Was which has properties of the astral sea: the laws of physics are only suggestions, spatial distribution makes no sense (think of the city scene in Inception), and it’s as if the island itself is dreaming of the former glory of Xerxes. In the astral sea geographic equivalent, Xerxes is eternally frozen in time and space in the moment that they fell into the astral sea. Viewed as one of the greatest losses in the failure of magic.
    Aurelius: Village ruin that is no longer inhabited. Previous inhabitants were swindled and bamboozled by Aurelion Sol the giant space dragon deceiver. They were promised eternal life in the Astral Sea if they committed mass suicide (under the guise of some ritual or what have you) but in the end, Aurelion Sol simply harvested their souls (sol sounds like soul, what a coincidence) and used them as a power source for his own magic and will in the astral sea.
    """
    ai = """Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence of humans and other animals. Example tasks in which this is done include speech recognition, computer vision, translation between (natural) languages, as well as other mappings of inputs.
    AI applications include advanced web search engines (e.g., Google Search), recommendation systems (used by YouTube, Amazon, and Netflix), understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Waymo and Tesla), generative or creative tools (ChatGPT and AI art), automated decision-making, and competing at the highest level in strategic game systems (such as chess and Go).[1]
    As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.[2]
    Artificial intelligence was founded as an academic discipline in 1956, and in the years since it has experienced several waves of optimism,[3][4] followed by disappointment and the loss of funding (known as an "AI winter"),[5][6] followed by new approaches, success, and renewed funding.[4][7] AI research has tried and discarded many different approaches, including simulating the brain, modeling human problem solving, formal logic, large databases of knowledge, and imitating animal behavior. In the first decades of the 21st century, highly mathematical and statistical machine learning has dominated the field, and this technique has proved highly successful, helping to solve many challenging problems throughout industry and academia.[7][8]
    The various sub-fields of AI research are centered around particular goals and the use of particular tools. The traditional goals of AI research include reasoning, knowledge representation, planning, learning, natural language processing, perception, and the ability to move and manipulate objects.[a] General intelligence (the ability to solve an arbitrary problem) is among the field's long-term goals.[9] To solve these problems, AI researchers have adapted and integrated a wide range of problem-solving techniques, including search and mathematical optimization, formal logic, artificial neural networks, and methods based on statistics, probability, and economics. AI also draws upon computer science, psychology, linguistics, philosophy, and many other fields.
    The field was founded on the assumption that human intelligence "can be so precisely described that a machine can be made to simulate it".[b] This raised philosophical arguments about the mind and the ethical consequences of creating artificial beings endowed with human-like intelligence; these issues have previously been explored by myth, fiction, and philosophy since antiquity.[11] Computer scientists and philosophers have since suggested that AI may become an existential risk to humanity if its rational capacities are not steered towards beneficial goals.[c] The term artificial intelligence has also been criticized for overhyping AI's true technological capabilities.[12][13][14]"""
    # filter info to be alphabetic only
    ai = "".join([c for c in ai if c.isalpha() or c == " "])

    roa = """The Roanoke Island, North Carolina, half dollar (also Roanoke Island half dollar) is a commemorative coin issued by the United States Bureau of the Mint in 1937. The coin commemorated the 350th anniversary of the Roanoke Colony, depicting Sir Walter Raleigh on one side, and on the other Eleanor Dare, holding her child, Virginia Dare, the first child of English descent born in an English colony in the Americas.
    The Roanoke Island half dollar was one of many commemorative issues authorized by Congress in 1936. Since it was intended to commemorate the 350th anniversary of the colony, founded in 1587, the coins were not struck until 1937. William Marks Simpson, a sculptor who created several commemorative coins of the era, designed the Roanoke Island issue. His work required only slight modification at the recommendation of the Commission of Fine Arts.
    The legislation allowed the Roanoke Memorial Association to buy at least 25,000 at a time, so long as the issue took place before July 1937, and the group placed two orders for the minimum amount. Eventually, 21,000 were returned to the Mint for redemption and melting. The Roanoke Island issue catalogs in the low hundreds of dollars."""
    roa = "".join([c for c in roa if c.isalpha() or c == " "])

    dance = """It seemed as if the palace had always housed the Atrius Building Commission, the company of clerks and estate agents who authored and notarized nearly every construction of any note in the Empire. It had stood for two hundred and fifty years, since the reign of the Emperor Magnus, a plain-fronted and austere hall on a minor but respectable plaza in the Imperial City. Energetic and ambitious middle-class lads and ladies worked there, as well as complacent middle-aged ones like Decumus Scotti. No one could imagine a world without the Commission, least of all Scotti. To be accurate, he could not imagine a world without himself in the Commission. "Lord Atrius is perfectly aware of your contributions," said the managing clerk, closing the shutter that demarcated Scotti's office behind him. "But you know that things have been difficult." "Yes," said Scotti, stiffly. "Lord Vanech's men have been giving us a lot of competition lately, and we must be more efficient if we are to survive. Unfortunately, that means releasing some of our historically best but presently underachieving senior clerks." "I understand. Can't be helped." "I'm glad that you understand," smiled the managing clerk, smiling thinly and withdrawing. "Please have your room cleared immediately." Scotti began the task of organizing all his work to pass on to his successor. It would probably be young Imbrallius who would take most of it on, which was as it should be, he considered philosophically. The lad knew how to find business. Scotti wondered idly what the fellow would do with the contracts for the new statue of St Alessia for which the Temple of the One had applied. Probably invent a clerical error, blame it on his old predecessor Decumus Scotti, and require an additional cost to rectify. "I have correspondence for Decumus Scotti of the Atrius Building Commission." Scotti looked up. A fat-faced courier had entered his office and was thrusting forth a sealed scroll. He handed the boy a gold piece, and opened it up. By the poor penmanship, atrocious spelling and grammar, and overall unprofessional tone, it was manifestly evident who the writer was. Liodes Jurus, a fellow clerk some years before, who had left the Commission after being accused of unethical business practices. "Dear Sckotti, I emagine you alway wondered what happened to me, and the last plase you would have expected to find me is out in the woods. But thats exactly where I am. Ha ha. If your'e smart and want to make lot of extra gold for Lord Atrius (and yourself, ha ha), youll come down to Vallinwood too. If you have'nt or have been following the politics hear lately, you may or may not know that ther's bin a war between the Boshmer and there neighbors Elswere over the past two years. Things have only just calm down, and ther's a lot that needs to be rebuilt. Now Ive got more business than I can handel, but I need somone with some clout, someone representing a respected agencie to get the quill in the ink. That somone is you, my fiend. Come & meat me at the M'ther Paskos Tavern in Falinnesti, Vallinwood. Ill be here 2 weeks and you wont be sorrie. -- Jurus P.S.: Bring a wagenload of timber if you can." "What do you have there, Scotti?" asked a voice. Scotti started. It was Imbrallius, his damnably handsome face peeking through the shutters, smiling in that way that melted the hearts of the stingiest of patrons and the roughest of stonemasons. Scotti shoved the letter in his jacket pocket. "Personal correspondence," he sniffed. "I'll be cleared up here in a just a moment." "I don't want to hurry you," said Imbrallius, grabbing a few sheets of blank contracts from Scotti's desk. "I've just gone through a stack, and the junior scribes hands are all cramping up, so I thought you wouldn't miss a few." The lad vanished. Scotti retrieved the letter and read it again. He thought about his life, something he rarely did. It seemed a sea of gray with a black insurmountable wall looming. There was only one narrow passage he could see in that wall. Quickly, before he had a moment to reconsider it, he grabbed a dozen of the blank contracts with the shimmering gold leaf ATRIUS BUILDING COMMISSION BY APPOINTMENT OF HIS IMPERIAL MAJESTY and hid them in the satchel with his personal effects. The next day he began his adventure with a giddy lack of hesitation. He arranged for a seat in a caravan bound for Valenwood, the single escorted conveyance to the southeast leaving the Imperial City that week. He had scarcely hours to pack, but he remembered to purchase a wagonload of timber. "It will be extra gold to pay for a horse to pull that," frowned the convoy head. "So I anticipated," smiled Scotti with his best Imbrallius grin. Ten wagons in all set off that afternoon through the familiar Cyrodilic countryside. Past fields of wildflowers, gently rolling woodlands, friendly hamlets. The clop of the horses' hooves against the sound stone road reminded Scotti that the Atrius Building Commission constructed it. Five of the eighteen necessary contracts for its completion were drafted by his own hand. "Very smart of you to bring that wood along," said a gray-whiskered Breton man next to him on his wagon. "You must be in Commerce." "Of a sort," said Scotti, in a way he hoped was mysterious, before introducing himself: "Decumus Scotti." "Gryf Mallon," said the man. "I'm a poet, actually a translator of old Bosmer literature. I was researching some newly discovered tracts of the Mnoriad Pley Bar two years ago when the war broke out and I had to leave. You are no doubt familiar with the Mnoriad, if you're aware of the Green Pact." Scotti thought the man might be speaking perfect gibberish, but he nodded his head. "Naturally, I don't pretend that the Mnoriad is as renowned as the Meh Ayleidion, or as ancient as the Dansir Gol, but I think it has a remarkable significance to understanding the nature of the merelithic Bosmer mind. The origin of the Wood Elf aversion to cutting their own wood or eating any plant material at all, yet paradoxically their willingness to import plantstuff from other cultures, I feel can be linked to a passage in the Mnoriad," Mallon shuffled through some of his papers, searching for the appropriate text. To Scotti's vast relief, the carriage soon stopped to camp for the night. They were high on a bluff over a gray stream, and before them was the great valley of Valenwood. Only the cry of seabirds declared the presence of the ocean to the bay to the west: here the timber was so tall and wide, twisting around itself like an impossible knot begun eons ago, to be impenetrable. A few more modest trees, only fifty feet to the lowest branches, stood on the cliff at the edge of camp. The sight was so alien to Scotti and he found himself so anxious about the proposition of entering the wilderness that he could not imagine sleeping. Fortunately, Mallon had supposed he had found another academic with a passion for the riddles of ancient cultures. Long into the night, he recited Bosmer verse in the original and in his own translation, sobbing and bellowing and whispering wherever appropriate. Gradually, Scotti began to feel drowsy, but a sudden crack of wood snapping made him sit straight up. "What was that?" Mallon smiled: "I like it too. 'Convocation in the malignity of the moonless speculum, a dance of fire --'" "There are some enormous birds up in the trees moving around," whispered Scotti, pointing in the direction of the dark shapes above. "I wouldn't worry about that," said Mallon, irritated with his audience. "Now listen to how the poet characterizes Herma-Mora's invocation in the eighteenth stanza of the fourth book." The dark shapes in the trees were some of them perched like birds, others slithered like snakes, and still others stood up straight like men. As Mallon recited his verse, Scotti watched the figures softly leap from branch to branch, half-gliding across impossible distances for anything without wings. They gathered in groups and then reorganized until they had spread to every tree around the camp. Suddenly they plummeted from the heights. "Mara!" cried Scotti. "They're falling like rain!" "Probably seed pods," Mallon shrugged, not turning around. "Some of the trees have remarkable --" The camp erupted into chaos. Fires burst out in the wagons, the horses wailed from mortal blows, casks of wine, fresh water, and liquor gushed their contents to the ground. A nimble shadow dashed past Scotti and Mallon, gathering sacks of grain and gold with impossible agility and grace. Scotti had only one glance at it, lit up by a sudden nearby burst of flame. It was a sleek creature with pointed ears, wide yellow eyes, mottled pied fur and a tail like a whip. "Werewolf," he whimpered, shrinking back. "Cathay-raht," groaned Mallon. "Much worse. Khajiti cousins or some such thing, come to plunder." "Are you sure?" As quickly as they struck, the creatures retreated, diving off the bluff before the battlemage and knight, the caravan's escorts, had fully opened their eyes. Mallon and Scotti ran to the precipice and saw a hundred feet below the tiny figures dash out of the water, shake themselves, and disappear into the wood. "Werewolves aren't acrobats like that," said Mallon. "They were definitely Cathay-raht. Bastard thieves. Thank Stendarr they didn't realize the value of my notebooks. It wasn't a complete loss."
    """

    info = roa
    stats = get_statistics(info)
    title = get_title(stats)
    subsections = get_subsections(stats, title, 6)
    section = write_section(stats, title, subsections[3])
    # format the title, subsections, sections, and stats
    print(f"Statistics: {stats}")
    print(f"Title: {title}")
    print(f"Subsections: {subsections}")
    print(f"Section: {section}")
