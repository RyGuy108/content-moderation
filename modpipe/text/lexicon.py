"""Gazetteers and term lists.

Two uses:
  * The classifier replaces gazetteer hits with placeholder tokens (<person>, <brand>, ...), so
    it learns the *context* that makes a mention risky ("<person> topless", "<brand> logo on a
    hoodie") instead of memorizing names. Adding a new protected name is a list edit, not a
    retrain.
  * The keyword-blocklist baseline in bench/ uses BLOCKLIST, the same terms a typical
    "first version" filter would ship with.

Ambiguous entries (a person named "Jordan", the brand "Apple") get their own placeholder so the
model can learn that "apple pie" is fine and "apple logo" is not.
"""

PUBLIC_FIGURES = [
    # politicians / heads of state
    "donald trump", "joe biden", "kamala harris", "barack obama", "michelle obama", "hillary clinton",
    "bernie sanders", "alexandria ocasio-cortez", "nancy pelosi", "mitch mcconnell", "ron desantis",
    "gavin newsom", "jd vance", "vladimir putin", "volodymyr zelensky", "xi jinping", "kim jong un",
    "emmanuel macron", "justin trudeau", "rishi sunak", "keir starmer", "narendra modi",
    "benjamin netanyahu", "angela merkel", "boris johnson", "king charles", "prince harry",
    "meghan markle", "kate middleton", "pope francis", "pope leo",
    # tech / business
    "elon musk", "mark zuckerberg", "jeff bezos", "bill gates", "tim cook", "sundar pichai",
    "sam altman", "jensen huang", "satya nadella", "warren buffett", "steve jobs", "jack ma",
    "oprah winfrey", "martha stewart", "mrbeast", "jimmy donaldson",
    # actors
    "tom cruise", "tom hanks", "brad pitt", "angelina jolie", "leonardo dicaprio", "margot robbie",
    "scarlett johansson", "jennifer lawrence", "jennifer aniston", "zendaya", "timothee chalamet",
    "keanu reeves", "dwayne johnson", "the rock", "will smith", "denzel washington", "morgan freeman",
    "emma watson", "emma stone", "sydney sweeney", "anya taylor-joy", "pedro pascal", "ryan reynolds",
    "chris hemsworth", "robert downey jr", "johnny depp", "amber heard", "meryl streep",
    "natalie portman", "gal gadot", "jenna ortega", "millie bobby brown", "florence pugh",
    "henry cavill", "keira knightley", "nicole kidman", "anne hathaway", "jason momoa",
    "idris elba", "cillian murphy", "ana de armas", "jacob elordi", "austin butler",
    # music
    "taylor swift", "beyonce", "rihanna", "ariana grande", "billie eilish", "selena gomez",
    "lady gaga", "dua lipa", "olivia rodrigo", "sabrina carpenter", "kanye west", "kim kardashian",
    "kylie jenner", "kendall jenner", "khloe kardashian", "nicki minaj", "cardi b", "doja cat",
    "justin bieber", "harry styles", "ed sheeran", "bad bunny", "travis scott", "eminem",
    "snoop dogg", "jay-z", "post malone", "the weeknd", "shakira", "jennifer lopez", "miley cyrus",
    "britney spears", "katy perry", "lizzo", "sza", "chappell roan", "charli xcx", "megan thee stallion",
    "ice spice", "lana del rey", "adele", "bruno mars", "kendrick lamar",
    # sports
    "lebron james", "stephen curry", "michael jordan", "kobe bryant", "cristiano ronaldo",
    "lionel messi", "neymar", "kylian mbappe", "serena williams", "simone biles", "tom brady",
    "patrick mahomes", "travis kelce", "shohei ohtani", "tiger woods", "usain bolt", "caitlin clark",
    "conor mcgregor", "naomi osaka", "roger federer", "rafael nadal", "novak djokovic",
    "lewis hamilton", "max verstappen", "virat kohli",
    # creators / media
    "logan paul", "jake paul", "kai cenat", "ishowspeed", "pewdiepie", "addison rae",
    "charli damelio", "khaby lame", "joe rogan", "tucker carlson", "greta thunberg", "andrew tate",
]

# Single-word or common-word names: the placeholder is <person_ambig> and context decides.
AMBIGUOUS_FIGURES = [
    "drake", "madonna", "prince", "pink", "cher", "eminem", "adele", "jordan", "washington",
    "lincoln", "kennedy", "obama", "trump", "biden", "putin", "musk", "swift", "beyonce", "rihanna",
    "zendaya", "shakira", "messi", "ronaldo", "lebron", "oprah", "kanye", "ye",
]

BRANDS = [
    "nike", "adidas", "puma", "reebok", "new balance", "under armour", "lululemon", "supreme",
    "gucci", "louis vuitton", "chanel", "prada", "versace", "balenciaga", "hermes", "dior",
    "rolex", "omega", "tiffany", "cartier", "coca-cola", "coke", "pepsi", "sprite", "fanta",
    "red bull", "monster energy", "starbucks", "mcdonalds", "burger king", "kfc", "taco bell",
    "wendys", "subway", "dominos", "pizza hut", "chipotle", "dunkin", "heineken", "budweiser",
    "corona", "apple", "iphone", "macbook", "google", "microsoft", "samsung", "sony",
    "playstation", "xbox", "nintendo", "tesla", "ferrari", "lamborghini", "porsche", "bmw",
    "mercedes", "audi", "toyota", "ford", "jeep", "harley-davidson", "netflix", "disney",
    "pixar", "marvel", "dc comics", "warner bros", "amazon", "walmart", "target", "ikea",
    "lego", "barbie", "hot wheels", "spotify", "tiktok", "instagram", "youtube", "facebook",
    "twitter", "openai", "chatgpt", "visa", "mastercard", "shell", "dove", "jaguar", "mustang",
    "oreo", "doritos", "pringles", "kitkat", "m&ms", "skittles", "hello kitty", "pokemon",
]

# Brand names that are also everyday words.
AMBIGUOUS_BRANDS = [
    "apple", "amazon", "target", "shell", "dove", "jaguar", "mustang", "corona", "subway", "puma",
    "monster", "visa", "ford", "sprite", "oreo", "tide", "gap", "orange", "fossil", "mini",
    "chase", "delta", "oracle", "pandora", "polo", "converse", "vans", "crocs", "patagonia",
    "mercedes", "lincoln", "jordan", "supreme", "prime", "dawn", "bounty", "arm & hammer",
]

CHARACTERS = [
    "mickey mouse", "minnie mouse", "donald duck", "spider-man", "spiderman", "batman", "superman",
    "wonder woman", "iron man", "captain america", "hulk", "thor", "deadpool", "wolverine",
    "darth vader", "yoda", "baby yoda", "grogu", "pikachu", "mario", "luigi", "sonic the hedgehog",
    "elsa", "shrek", "buzz lightyear", "woody", "harry potter", "hermione", "homer simpson",
    "bart simpson", "spongebob", "patrick star", "peppa pig", "bluey", "hello kitty", "snoopy",
    "garfield", "winnie the pooh", "stitch", "simba", "goku", "naruto", "luffy", "kirby",
    "link from zelda", "master chief", "lara croft", "optimus prime", "barbie", "ken doll",
    "scooby-doo", "bugs bunny", "the minions", "minion", "olaf", "moana", "rapunzel", "ariel",
]

# Terms a typical keyword blocklist ships with (used by the baseline and as weak features).
BLOCKLIST = {
    "sexual": ["nude", "naked", "nudity", "topless", "porn", "porno", "pornographic", "nsfw",
               "xxx", "sex", "sexual", "explicit", "genitals", "nipples", "erotic", "hentai",
               "orgasm", "fetish", "bdsm", "onlyfans", "boobs", "tits", "penis", "vagina",
               "strip", "stripper", "lewd", "horny", "masturbating", "intercourse"],
    "minor_sexual": ["child porn", "cp", "loli", "lolicon", "shota", "underage", "jailbait",
                     "preteen"],
    "gore": ["gore", "decapitated", "decapitation", "dismembered", "disemboweled", "beheaded",
             "beheading", "mutilated", "severed head", "guts", "corpse", "entrails", "bloodbath",
             "massacre", "torture", "snuff"],
    "hate": ["nazi", "swastika", "kkk", "white power", "heil hitler", "subhuman", "vermin",
             "ethnic cleansing", "genocide", "14 words", "racial slur", "white supremacy",
             "inferior race"],
    "self_harm": ["suicide", "self harm", "self-harm", "cutting", "kill myself", "kill yourself",
                  "noose", "slit wrists", "overdose", "thinspo", "thinspiration", "pro ana",
                  "pro-ana", "kys"],
    "real_person": ["deepfake", "face swap", "lookalike", "impersonate", "celebrity"],
    "trademark": ["logo", "trademark", "branded", "brand logo"],
}

LABELS = ["sexual", "minor_sexual", "gore", "hate", "self_harm", "real_person", "trademark"]

# Extra concept lists the classifier sees as <lex_*> features (not part of the baseline's
# blocklist, because they are not block-worthy on their own: "teen" and "lingerie" are fine
# separately, the combination is not).
CONCEPTS = {
    "minor": ["child", "children", "kid", "kids", "teen", "teens", "teenage", "teenager", "minor",
              "schoolgirl", "schoolboy", "little girl", "little boy", "young girl", "young boy",
              "toddler", "preteen", "underage", "year old", "years old", "yo", "middle schooler",
              "high schooler", "freshman", "school uniform", "baby", "infant", "boy", "girl",
              "loli", "shota", "jailbait", "barely legal", "young teen", "tween"],
    "suggestive": ["lingerie", "sexy", "seductive", "bikini", "lewd", "sensual", "see-through",
                   "provocative", "underwear", "revealing", "panties", "thong", "bra", "skimpy",
                   "erotic", "flirty", "posing", "bending over", "spread", "naughty", "steamy"],
    "body": ["breasts", "boobs", "tits", "nipples", "nips", "butt", "ass", "crotch", "genitals",
             "penis", "vagina", "thighs", "sideboob", "cleavage", "chest"],
    "violence": ["blood", "bloody", "kill", "killed", "dead", "dying", "stab", "stabbed", "shot",
                 "murder", "corpse", "body", "wound", "gun", "knife", "bleeding", "torture"],
}
