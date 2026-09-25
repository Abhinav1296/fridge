"""Cook Mode — a guided, step-by-step cooking walkthrough for a single recipe.

The bundled recipe corpus (``data/recipes.json``) records *what* goes into each dish
(ingredients, tags, time) but not *how* to cook it. Rather than ship a second hand-written
instructions file that could drift out of sync, this module **synthesises** a sensible,
method-aware walkthrough from the structured data the app already trusts:

* the recipe's **type tag** (curry, dal, dry-sabzi, rice, soup, grill, drink, …) selects a
  cooking *method*, and for the tag-less breakfast/quick dishes the **hero ingredient**
  (egg, besan, poha, tofu, …) does the same;
* the recipe's **ingredients** are sorted into culinary *roles* (aromatics, tomato base,
  dairy, greens, garnish, …) so each step names the real things in front of the cook;
* the recipe's **time_min** bounds the per-step timers so the suggested durations add up to
  roughly the stated cook time.

On top of the walkthrough, Cook Mode surfaces **swap ideas**: for every non-pantry
ingredient it asks the trained ingredient embeddings (:func:`recommender.similar_ingredients`)
for the nearest alternatives, so a cook who is out of one thing sees what behaves like it in
other recipes. Substitutions are best-effort suggestions from co-occurrence vectors, never a
guarantee, and the UI labels them as such.

Like the rest of the recommendation engine this module is **framework-agnostic** (no Flask,
no HTTP) and reads only the standard library plus :mod:`recommender`, so the web layer or the
autonomous agent can call :func:`build_cook_session` directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import recommender

# --- Ingredient roles --------------------------------------------------------
# Canonical recipe tokens grouped by the job they do in a dish. Membership drives both
# which step an ingredient is mentioned in and how the walkthrough reads. Kept as frozensets
# so lookups are cheap and the tables read as data, not code.

_AROMATICS = frozenset({"onion", "garlic", "ginger", "green_chili"})
_TOMATO = frozenset({"tomato"})
_RICH_DAIRY = frozenset({"butter", "cream"})
_YOGURT = frozenset({"yogurt"})
_CHEESE = frozenset({"cheese"})
_MILK = frozenset({"milk"})
_GREENS = frozenset({
    "spinach", "peas", "capsicum", "beans", "carrot", "cabbage", "cauliflower",
    "potato", "okra", "eggplant", "mushroom", "broccoli", "corn", "cucumber",
})
_GARNISH = frozenset({"coriander", "lemon", "mint"})
_FRUIT = frozenset({"banana", "mango", "apple", "orange", "grapes"})
_SWEETENER = frozenset({"honey", "sugar"})
# Animal-derived tokens, filtered out of swap ideas for a vegetarian dish so a paneer curry
# never suggests "chicken" as an alternative.
_NON_VEG = frozenset({"chicken", "egg", "mutton", "fish", "prawn"})

# Type tags that select a cooking method, in priority order (first match wins).
_METHOD_TAGS = ("drink", "soup", "grill", "rice", "dal", "dry-sabzi", "curry")


# --- Small text helpers ------------------------------------------------------


def _present(tokens: Sequence[str], role: Iterable[str]) -> list[str]:
    """Recipe tokens that fall in ``role``, preserving the recipe's own ordering."""
    role_set = set(role)
    return [t for t in tokens if t in role_set]


def _name(token: str) -> str:
    """Human label for a token ('green_chili' → 'Green chili')."""
    return recommender._prettify(token)


def _phrase(tokens: Sequence[str], *, conj: str = "and") -> str:
    """Join tokens into a lower-case list for mid-sentence use ('onion, tomato and garlic')."""
    names = [_name(t).lower() for t in tokens]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} {conj} {names[1]}"
    return f"{', '.join(names[:-1])} {conj} {names[-1]}"


def _step(title: str, text: str, uses: Sequence[str], seconds: int = 0) -> dict[str, Any]:
    """Assemble one walkthrough step, prettifying the ingredient chips."""
    return {
        "title": title,
        "text": text,
        "seconds": int(seconds),
        "uses": [_name(t) for t in uses],
    }


# --- Method builders ---------------------------------------------------------
# Each builder receives the recipe's ingredient tokens (hero first) and its cook time in
# minutes, and returns an ordered list of steps. They lean on the role helpers so the same
# builder reads naturally across every recipe that shares a method.


def _steps_curry(ings: list[str], minutes: int) -> list[dict[str, Any]]:
    hero = ings[0]
    aromatics = _present(ings, _AROMATICS)
    tomato = _present(ings, _TOMATO)
    rich = _present(ings, _RICH_DAIRY)
    yogurt = _present(ings, _YOGURT)
    greens = [t for t in _present(ings, _GREENS) if t != hero]
    garnish = _present(ings, _GARNISH)
    fat = "butter" if "butter" in rich else "oil"
    simmer = max(240, minutes * 60 - 720)

    steps = [_step(
        "Prep",
        f"Finely chop {_phrase(aromatics + tomato) or 'the aromatics'}"
        + (f", and cube the {_name(hero).lower()}" if hero not in _GREENS else "")
        + ". Keep everything within reach before the pan gets hot.",
        aromatics + tomato + [hero],
    )]
    onion = [t for t in aromatics if t == "onion"]
    gg = [t for t in aromatics if t in {"ginger", "garlic", "green_chili"}]
    if onion:
        steps.append(_step(
            "Sweat the onion",
            f"Heat 1–2 tbsp {fat} in a pan and sauté the onion until soft and golden.",
            onion, seconds=300,
        ))
    if gg:
        steps.append(_step(
            "Bloom aromatics & spice",
            f"Add {_phrase(gg)} and cook 1 minute, then stir in turmeric, chilli powder and "
            "garam masala (pantry) with a pinch of salt until fragrant.",
            gg, seconds=90,
        ))
    else:
        steps.append(_step(
            "Spice it",
            "Stir in turmeric, chilli powder and garam masala (pantry) with a pinch of salt.",
            [], seconds=60,
        ))
    if tomato:
        steps.append(_step(
            "Build the masala",
            f"Add the {_phrase(tomato)} and cook down, stirring, until the oil separates and "
            "the base turns thick and glossy.",
            tomato, seconds=360,
        ))
    if yogurt:
        steps.append(_step(
            "Enrich",
            "Lower the heat and whisk in the yogurt a spoon at a time so it stays smooth.",
            yogurt, seconds=90,
        ))
    steps.append(_step(
        "Add the star",
        f"Fold in the {_phrase([hero] + greens)}"
        + ("" if hero in _GREENS else "")
        + ". Pour in a splash of water for a gravy and bring to a gentle bubble.",
        [hero] + greens, seconds=90,
    ))
    steps.append(_step(
        "Simmer",
        f"Cover and simmer until the {_name(hero).lower()} is tender and the flavours meld, "
        "stirring once or twice.",
        [hero], seconds=simmer,
    ))
    if "cream" in ings:
        steps.append(_step(
            "Finish rich",
            "Swirl in the cream off the heat for a velvety gravy.",
            ["cream"], seconds=30,
        ))
    steps.append(_step(
        "Serve",
        "Taste and adjust salt"
        + (f", then garnish with {_phrase(garnish)}" if garnish else "")
        + ". Serve hot with rice or roti.",
        garnish,
    ))
    return steps


def _steps_dal(ings: list[str], minutes: int) -> list[dict[str, Any]]:
    lentil = _present(ings, {"lentil"}) or [ings[0]]
    aromatics = _present(ings, _AROMATICS)
    tomato = _present(ings, _TOMATO)
    greens = _present(ings, _GREENS)
    garnish = _present(ings, _GARNISH)
    return [
        _step("Boil the lentils",
              f"Rinse the {_phrase(lentil)} well, then pressure-cook or boil with turmeric, "
              "salt and plenty of water until soft and collapsing.",
              lentil, seconds=min(minutes * 60 - 360, 720) if minutes > 12 else 480),
        _step("Start the tadka",
              "Heat ghee or oil and crackle cumin (pantry), then add "
              f"{_phrase(aromatics) or 'the aromatics'} and fry until golden.",
              aromatics, seconds=240),
        *([_step("Tomato base",
                 f"Add the {_phrase(tomato)} and cook to a soft pulp.",
                 tomato, seconds=180)] if tomato else []),
        *([_step("Wilt the greens",
                 f"Stir in the {_phrase(greens)} until wilted.",
                 greens, seconds=120)] if greens else []),
        _step("Combine",
              "Pour the tempering into the cooked dal, loosen with water to the consistency "
              "you like, and season.",
              [], seconds=60),
        _step("Simmer",
              "Simmer a few minutes so the tadka flavours soak in.",
              [], seconds=300),
        _step("Serve",
              "Finish with a squeeze of lemon"
              + (f" and {_phrase(garnish)}" if garnish else "")
              + ". Serve with rice.",
              garnish),
    ]


def _steps_dry_sabzi(ings: list[str], minutes: int) -> list[dict[str, Any]]:
    hero = ings[0]
    aromatics = _present(ings, _AROMATICS)
    tomato = _present(ings, _TOMATO)
    veg = [t for t in _present(ings, _GREENS) if t != hero]
    garnish = _present(ings, _GARNISH)
    return [
        _step("Prep",
              f"Wash and cut the {_phrase([hero] + veg)} into even pieces so they cook at the "
              "same rate.",
              [hero] + veg),
        _step("Temper",
              "Heat oil, splutter cumin and mustard seeds (pantry), then add "
              f"{_phrase(aromatics) or 'the onion'}.",
              aromatics, seconds=180),
        *([_step("Soften the tomato",
                 f"Add the {_phrase(tomato)} and cook briefly.",
                 tomato, seconds=120)] if tomato else []),
        _step("Add the veg",
              f"Tip in the {_phrase([hero] + veg)} with turmeric, chilli powder and salt; "
              "toss to coat.",
              [hero] + veg, seconds=60),
        _step("Cook through",
              "Cover and cook on medium, stirring now and then, until just tender — keep it dry, "
              "not mushy.",
              [], seconds=max(240, minutes * 60 - 360)),
        _step("Serve",
              "Finish with"
              + (f" {_phrase(garnish)} and" if garnish else "")
              + " a drizzle of lemon. Great with roti.",
              garnish),
    ]


def _steps_rice(ings: list[str], minutes: int) -> list[dict[str, Any]]:
    aromatics = _present(ings, _AROMATICS)
    veg = [t for t in _present(ings, _GREENS)]
    protein = _present(ings, {"chicken", "egg", "paneer", "tofu"})
    yogurt = _present(ings, _YOGURT)
    garnish = _present(ings, _GARNISH)
    # Curd rice and jeera rice are gentle, near-plain preparations.
    if yogurt:
        return [
            _step("Cook & cool the rice",
                  "Cook the rice soft, spread it out and let it cool to just warm.",
                  ["rice"], seconds=600),
            _step("Fold in yogurt",
                  "Mash lightly and mix in the yogurt (and a splash of milk) to a creamy set.",
                  yogurt, seconds=60),
            _step("Temper & serve",
                  "Pour over a tadka of mustard seeds, curry leaves and green chilli (pantry). "
                  "Serve chilled or at room temperature.",
                  [], seconds=60),
        ]
    return [
        _step("Rinse the rice",
              "Rinse the rice until the water runs clear, then soak for 10 minutes and drain.",
              ["rice"]),
        _step("Temper",
              "Heat oil or ghee, add cumin and whole spices (pantry)"
              + (f", then {_phrase(aromatics)}" if aromatics else "") + ".",
              aromatics, seconds=180),
        *([_step("Sear the protein",
                 f"Add the {_phrase(protein)} and cook until it changes colour.",
                 protein, seconds=240)] if protein else []),
        *([_step("Add the veg",
                 f"Stir in the {_phrase(veg)} and sauté a couple of minutes.",
                 veg, seconds=150)] if veg else []),
        _step("Add rice & water",
              "Add the drained rice with twice its volume of water and salt; stir once.",
              ["rice"], seconds=60),
        _step("Cook covered",
              "Cover and cook on low until the water is absorbed and the grains are fluffy.",
              [], seconds=max(480, minutes * 60 - 480)),
        _step("Rest & fluff",
              "Rest 5 minutes off the heat, then fluff with a fork"
              + (f" and finish with {_phrase(garnish)}" if garnish else "") + ".",
              garnish, seconds=300),
    ]


def _steps_soup(ings: list[str], minutes: int) -> list[dict[str, Any]]:
    aromatics = _present(ings, _AROMATICS)
    tomato = _present(ings, _TOMATO)
    veg = [t for t in _present(ings, _GREENS)]
    rich = _present(ings, _RICH_DAIRY)
    body = tomato + veg
    return [
        _step("Sauté the base",
              f"Melt {'butter' if 'butter' in rich else 'a little oil'} and soften "
              f"{_phrase(aromatics) or 'the aromatics'}.",
              aromatics, seconds=180),
        _step("Add the body",
              f"Add the {_phrase(body) or 'vegetables'} and cook until slumped.",
              body, seconds=300),
        _step("Simmer",
              "Pour in water or stock, season, and simmer until everything is very soft.",
              [], seconds=max(300, minutes * 60 - 480)),
        _step("Blend",
              "Blend smooth (careful with the hot liquid) and pass back into the pan; loosen "
              "with water if needed.",
              [], seconds=60),
        _step("Finish",
              "Reheat, adjust salt and pepper"
              + (", swirl in the cream" if "cream" in ings else "")
              + ", and serve hot.",
              _present(ings, {"cream"})),
    ]


def _steps_grill(ings: list[str], minutes: int) -> list[dict[str, Any]]:
    hero = ings[0]
    veg = [t for t in _present(ings, _GREENS) if t != hero]
    yogurt = _present(ings, _YOGURT)
    return [
        _step("Make the marinade",
              f"Whisk the {_phrase(yogurt) or 'yogurt'} with ginger-garlic paste, chilli, "
              "turmeric, garam masala and salt (pantry) into a thick coating.",
              yogurt, seconds=120),
        _step("Coat",
              f"Cube the {_phrase([hero] + veg)} and turn through the marinade until well "
              "covered.",
              [hero] + veg, seconds=120),
        _step("Marinate",
              "Cover and rest so the flavour soaks in — longer is better.",
              [], seconds=min(minutes * 60 - 600, 1200) if minutes > 15 else 900),
        _step("Grill",
              "Thread onto skewers and grill, air-fry or pan-sear on high, turning, until "
              "charred at the edges.",
              [], seconds=600),
        _step("Serve",
              "Dust with chaat masala and a squeeze of lemon; serve hot with mint chutney.",
              []),
    ]


def _steps_drink(ings: list[str], minutes: int) -> list[dict[str, Any]]:
    fruit = _present(ings, _FRUIT)
    milk = _present(ings, _MILK)
    yogurt = _present(ings, _YOGURT)
    sweet = _present(ings, _SWEETENER)
    liquid = yogurt + milk
    return [
        _step("Prep the fruit",
              f"Peel and roughly chop the {_phrase(fruit) or 'fruit'}.",
              fruit),
        _step("Blend",
              f"Blend the fruit with the {_phrase(liquid) or 'milk'}"
              + (f" and {_phrase(sweet)}" if sweet else "")
              + " until completely smooth.",
              liquid + sweet, seconds=60),
        _step("Serve",
              "Add a few ice cubes, blend once more, pour into a tall glass and serve cold.",
              []),
    ]


# --- Hero-ingredient specials (tag-less breakfast / quick dishes) ------------


def _steps_egg(ings: list[str], minutes: int) -> list[dict[str, Any]]:
    aromatics = _present(ings, _AROMATICS | _TOMATO)
    garnish = _present(ings, _GARNISH)
    return [
        _step("Beat & chop",
              f"Beat the eggs with salt and pepper, and finely chop "
              f"{_phrase(aromatics + garnish) or 'the aromatics'}.",
              ["egg"] + aromatics + garnish),
        _step("Sauté aromatics",
              f"Heat a little oil and soften {_phrase(aromatics) or 'the onion'} with a pinch "
              "of turmeric and chilli.",
              aromatics, seconds=180),
        _step("Cook the egg",
              "Pour in the beaten egg. For an omelette let it set then fold; for bhurji keep "
              "stirring until scrambled and just set.",
              ["egg"], seconds=180),
        _step("Serve",
              "Slide onto a plate"
              + (f", scatter {_phrase(garnish)}" if garnish else "")
              + " and serve hot with toast.",
              garnish),
    ]


def _steps_scramble(ings: list[str], minutes: int) -> list[dict[str, Any]]:
    hero = ings[0]
    aromatics = _present(ings, _AROMATICS | _TOMATO)
    return [
        _step("Prep",
              f"Crumble the {_name(hero).lower()} and finely chop "
              f"{_phrase(aromatics) or 'the aromatics'}.",
              [hero] + aromatics),
        _step("Sauté",
              f"Heat oil and cook {_phrase(aromatics) or 'the onion'} with turmeric, chilli "
              "and salt (pantry) until soft.",
              aromatics, seconds=240),
        _step("Toss the star",
              f"Add the crumbled {_name(hero).lower()} and toss on high for a couple of minutes "
              "so it takes on the masala.",
              [hero], seconds=150),
        _step("Serve",
              "Finish with coriander and serve hot with bread or roti.",
              _present(ings, {"coriander"})),
    ]


def _steps_batter(ings: list[str], minutes: int) -> list[dict[str, Any]]:
    hero = ings[0]
    mixins = [t for t in ings if t != hero]
    return [
        _step("Make the batter",
              f"Whisk the {_name(hero).lower()} with water, salt and turmeric into a smooth, "
              "pourable batter with no lumps.",
              [hero], seconds=60),
        _step("Fold in the veg",
              f"Stir in finely chopped {_phrase(mixins) or 'vegetables'}.",
              mixins),
        _step("Rest",
              "Let the batter rest a few minutes so it thickens slightly.",
              [], seconds=300),
        _step("Cook",
              "Ladle onto a hot, lightly oiled tawa, spread into a thin round, and cook both "
              "sides until golden and crisp at the edges.",
              [], seconds=300),
        _step("Serve",
              "Serve hot off the pan with chutney or ketchup.",
              []),
    ]


def _steps_upma(ings: list[str], minutes: int) -> list[dict[str, Any]]:
    hero = ings[0]
    veg = [t for t in ings if t != hero]
    return [
        _step("Roast the semolina",
              f"Dry-roast the {_name(hero).lower()} on low until it smells nutty; set aside.",
              [hero], seconds=240),
        _step("Temper",
              "Heat oil, splutter mustard seeds and curry leaves (pantry), then add "
              f"{_phrase(veg) or 'the vegetables'} and sauté.",
              veg, seconds=180),
        _step("Add water",
              "Pour in about twice the volume of water with salt and bring to a boil.",
              [], seconds=120),
        _step("Cook",
              f"Lower the heat and rain in the roasted {_name(hero).lower()}, stirring "
              "constantly so it stays lump-free; cover and steam until fluffy.",
              [hero], seconds=180),
        _step("Serve",
              "Fluff, finish with a squeeze of lemon, and serve hot.",
              []),
    ]


def _steps_poha(ings: list[str], minutes: int) -> list[dict[str, Any]]:
    hero = ings[0]
    veg = [t for t in ings if t not in {hero, "lemon", "peanut"}]
    return [
        _step("Rinse the poha",
              "Rinse the poha in a colander until just soft, then leave to drain — don't let it "
              "go mushy.",
              [hero]),
        _step("Temper",
              "Heat oil, splutter mustard seeds (pantry), fry the peanuts, then add "
              f"{_phrase(veg) or 'the vegetables'} with turmeric and salt.",
              _present(ings, {"peanut"}) + veg, seconds=300),
        _step("Combine",
              "Fold in the drained poha gently until evenly coloured and heated through.",
              [hero], seconds=120),
        _step("Serve",
              "Turn off the heat, squeeze over the lemon, and serve warm.",
              _present(ings, {"lemon"})),
    ]


def _steps_porridge(ings: list[str], minutes: int) -> list[dict[str, Any]]:
    hero = ings[0]
    milk = _present(ings, _MILK)
    fruit = _present(ings, _FRUIT)
    sweet = _present(ings, _SWEETENER)
    return [
        _step("Simmer",
              f"Bring the {_phrase(milk) or 'milk'} (or water) to a gentle simmer and stir in "
              f"the {_name(hero).lower()}.",
              [hero] + milk, seconds=300),
        _step("Cook to creamy",
              "Cook, stirring, until thick and creamy; add more liquid if it gets too stiff.",
              [], seconds=120),
        _step("Sweeten & top",
              f"Sweeten with {_phrase(sweet) or 'honey'} and top with sliced "
              f"{_phrase(fruit) or 'fruit'}. Serve warm.",
              sweet + fruit),
    ]


def _steps_paratha(ings: list[str], minutes: int) -> list[dict[str, Any]]:
    filling = [t for t in ings if t != "flour"]
    return [
        _step("Knead the dough",
              "Knead the flour with a little salt and water into a soft, smooth dough; rest it "
              "covered.",
              ["flour"], seconds=600),
        _step("Make the filling",
              f"Mash the boiled {_phrase(filling) or 'potato'} with salt, chilli and "
              "spices (pantry) into a dry, well-seasoned filling.",
              filling),
        _step("Stuff & roll",
              "Take a ball of dough, tuck in a spoon of filling, seal, and roll out gently into "
              "a round, dusting with flour.",
              [], seconds=180),
        _step("Cook",
              "Cook on a hot tawa with a little ghee, pressing, until golden brown spots appear "
              "on both sides.",
              [], seconds=240),
        _step("Serve",
              "Serve hot with curd, pickle or butter.",
              []),
    ]


def _steps_noodles(ings: list[str], minutes: int) -> list[dict[str, Any]]:
    hero = ings[0]
    veg = [t for t in ings if t != hero]
    return [
        _step("Boil the noodles",
              f"Boil the {_name(hero).lower()} until just done, drain, and toss with a drop of "
              "oil so they don't stick.",
              [hero], seconds=360),
        _step("Stir-fry the veg",
              f"On high heat, stir-fry the finely shredded {_phrase(veg) or 'vegetables'} for a "
              "minute or two — keep them crunchy.",
              veg, seconds=180),
        _step("Toss together",
              "Add the noodles with soy sauce, salt and pepper (pantry) and toss everything "
              "over high heat until coated.",
              [hero], seconds=120),
        _step("Serve",
              "Serve immediately, while hot and glossy.",
              []),
    ]


def _steps_pasta(ings: list[str], minutes: int) -> list[dict[str, Any]]:
    aromatics = _present(ings, {"onion", "garlic"})
    tomato = _present(ings, _TOMATO)
    cheese = _present(ings, _CHEESE)
    return [
        _step("Boil the pasta",
              "Boil the pasta in salted water until al dente; save a little cooking water, then "
              "drain.",
              ["pasta"], seconds=600),
        _step("Make the sauce",
              f"Sauté {_phrase(aromatics) or 'the garlic'}, then add the {_phrase(tomato)} and "
              "cook into a thick sauce with salt and pepper.",
              aromatics + tomato, seconds=360),
        _step("Toss",
              "Fold the pasta through the sauce with a splash of the pasta water"
              + (f" and the {_phrase(cheese)}" if cheese else "") + ".",
              cheese, seconds=60),
        _step("Serve",
              "Serve hot"
              + (", with extra cheese on top" if cheese else "") + ".",
              cheese),
    ]


def _steps_stir_fry(ings: list[str], minutes: int) -> list[dict[str, Any]]:
    hero = ings[0]
    aromatics = _present(ings, {"garlic", "ginger", "onion"})
    veg = [t for t in _present(ings, _GREENS) if t != hero]
    return [
        _step("Prep",
              f"Press and cube the {_name(hero).lower()}; cut the {_phrase(veg) or 'vegetables'} "
              "into bite-size pieces.",
              [hero] + veg),
        _step("Sear the star",
              f"Get a pan very hot with a little oil and sear the {_name(hero).lower()} until "
              "golden on the edges; set aside.",
              [hero], seconds=240),
        _step("Stir-fry the veg",
              f"Stir-fry {_phrase(aromatics) or 'the garlic'} and the {_phrase(veg) or 'veg'} "
              "on high, keeping everything crisp.",
              aromatics + veg, seconds=180),
        _step("Combine & sauce",
              f"Return the {_name(hero).lower()}, add soy sauce (pantry) and toss to coat.",
              [hero], seconds=60),
        _step("Serve",
              "Serve hot over rice or noodles.",
              []),
    ]


def _steps_toastie(ings: list[str], minutes: int) -> list[dict[str, Any]]:
    fillings = [t for t in ings if t not in {"bread", "butter"}]
    return [
        _step("Prep the fillings",
              f"Slice the {_phrase(fillings) or 'fillings'} thinly and season with salt and "
              "pepper.",
              fillings),
        _step("Build",
              "Butter the bread on the outside and layer the fillings between two slices.",
              _present(ings, {"bread", "butter"})),
        _step("Toast",
              "Toast in a pan or sandwich press, pressing, until golden and crisp and any "
              "cheese has melted.",
              _present(ings, _CHEESE), seconds=240),
        _step("Serve",
              "Cut in half and serve hot with ketchup or chutney.",
              []),
    ]


def _steps_raita(ings: list[str], minutes: int) -> list[dict[str, Any]]:
    yogurt = _present(ings, _YOGURT)
    veg = [t for t in ings if t not in _YOGURT]
    return [
        _step("Whisk the yogurt",
              "Whisk the yogurt smooth with a little water, salt and roasted cumin (pantry).",
              yogurt, seconds=60),
        _step("Fold in the veg",
              f"Finely chop or grate the {_phrase(veg) or 'vegetables'} and fold them in.",
              veg),
        _step("Chill & serve",
              "Chill briefly and serve cold as a cooling side.",
              []),
    ]


def _steps_salad(ings: list[str], minutes: int) -> list[dict[str, Any]]:
    fruit = _present(ings, _FRUIT)
    other = [t for t in ings if t not in _FRUIT]
    core = fruit + other
    return [
        _step("Chop",
              f"Wash and chop the {_phrase(core) or 'ingredients'} into even bite-size pieces.",
              core),
        _step("Toss",
              "Toss together in a bowl with a squeeze of lemon and a pinch of salt or chaat "
              "masala.",
              [], seconds=0),
        _step("Serve",
              "Serve fresh and cold.",
              []),
    ]


def _steps_generic(ings: list[str], minutes: int) -> list[dict[str, Any]]:
    hero = ings[0]
    aromatics = _present(ings, _AROMATICS)
    tomato = _present(ings, _TOMATO)
    veg = [t for t in _present(ings, _GREENS) if t != hero]
    garnish = _present(ings, _GARNISH)
    rest = [t for t in ings if t not in set(aromatics + tomato + veg + garnish + [hero])]
    return [
        _step("Prep",
              f"Gather and prepare the {_phrase(ings)}; chop what needs chopping.",
              ings),
        _step("Sauté the base",
              f"Heat oil and cook {_phrase(aromatics) or 'the aromatics'}"
              + (f" and the {_phrase(tomato)}" if tomato else "")
              + " with salt and everyday spices (pantry).",
              aromatics + tomato, seconds=240),
        _step("Add the mains",
              f"Add the {_phrase([hero] + veg + rest)} and stir to combine.",
              [hero] + veg + rest, seconds=90),
        _step("Cook through",
              "Cook until everything is tender and well seasoned, adding a little water if it "
              "looks dry.",
              [], seconds=max(240, minutes * 60 - 330)),
        _step("Serve",
              "Taste, adjust the seasoning"
              + (f", garnish with {_phrase(garnish)}" if garnish else "")
              + " and serve hot.",
              garnish),
    ]


# Hero-token → builder, checked before the generic fallback for tag-less dishes.
_HERO_BUILDERS = {
    "egg": _steps_egg,
    "paneer": _steps_scramble,  # only reached for tag-less paneer_bhurji
    "besan": _steps_batter,
    "semolina": _steps_upma,
    "poha": _steps_poha,
    "oats": _steps_porridge,
    "flour": _steps_paratha,
    "noodles": _steps_noodles,
    "pasta": _steps_pasta,
    "tofu": _steps_stir_fry,
}


def _classify(recipe: dict[str, Any]) -> tuple[str, Any]:
    """Pick a ``(method_name, builder)`` for a recipe from its tags, then its hero token."""
    tags = {str(t).lower() for t in recipe.get("tags", [])}
    ings = [str(t).lower() for t in recipe.get("ingredients", [])]
    for tag in _METHOD_TAGS:
        if tag in tags:
            return tag, {
                "drink": _steps_drink, "soup": _steps_soup, "grill": _steps_grill,
                "rice": _steps_rice, "dal": _steps_dal, "dry-sabzi": _steps_dry_sabzi,
                "curry": _steps_curry,
            }[tag]
    hero = ings[0] if ings else ""
    if hero in _HERO_BUILDERS:
        return hero, _HERO_BUILDERS[hero]
    # Side / snack heuristics keyed on signature ingredients.
    if "bread" in ings:
        return "toastie", _steps_toastie
    if "yogurt" in ings and "cucumber" in ings:
        return "raita", _steps_raita
    if len(_present(ings, _FRUIT)) >= 2 or "cucumber" in ings and "yogurt" not in ings:
        return "salad", _steps_salad
    return "generic", _steps_generic


# --- Substitutions -----------------------------------------------------------


def _corpus_tokens() -> set[str]:
    """Every ingredient token used anywhere in the recipe corpus."""
    tokens: set[str] = set()
    for recipe in recommender.all_recipes():
        tokens.update(str(t).lower() for t in recipe.get("ingredients", []))
    return tokens


def _swap_ideas(
    token: str,
    *,
    own: set[str],
    exclude: set[str],
    candidates: set[str],
    k: int,
    min_similarity: float,
) -> list[dict[str, Any]]:
    """Embedding-nearest ingredients that could stand in for ``token``.

    Never suggests a pantry staple, the recipe's own ingredients, or an excluded
    (disliked/allergen) token. Empty when embeddings are unavailable.
    """
    pool = candidates - own - exclude - recommender.PANTRY - {token}
    ranked = recommender.similar_ingredients(
        token, k=k, candidates=pool, min_similarity=min_similarity
    )
    return [
        {"token": cand, "name": _name(cand), "similarity": round(sim, 3)}
        for cand, sim in ranked
    ]


# --- Public entry point ------------------------------------------------------


def build_cook_session(
    recipe: dict[str, Any],
    *,
    have_tokens: Iterable[str] = (),
    exclude_tokens: Iterable[str] = (),
    sub_k: int = 3,
    min_similarity: float = 0.35,
    servings: int = 2,
) -> dict[str, Any]:
    """Assemble a full Cook Mode session for one recipe.

    ``recipe`` is a corpus entry (id/title/ingredients/tags/time_min). ``have_tokens`` are the
    canonical tokens currently in the fridge (used to mark each ingredient have/missing);
    ``exclude_tokens`` are allergens or dislikes that must never appear as a swap idea. The
    result bundles the synthesised, method-aware steps, an ingredient checklist with per-item
    swap ideas, and light metadata for the UI.
    """
    ings = [str(t).lower() for t in recipe.get("ingredients", [])]
    minutes = int(recipe.get("time_min") or 20)
    have = {str(t).lower() for t in have_tokens}
    exclude = {str(t).lower() for t in exclude_tokens}
    tags = {str(t).lower() for t in recipe.get("tags", [])}
    # A vegetarian dish should never propose a meat/egg swap.
    if "non-veg" not in tags:
        exclude = exclude | _NON_VEG
    own = set(ings)
    candidates = _corpus_tokens()

    method, builder = _classify(recipe)
    steps = builder(ings, minutes)
    for i, step in enumerate(steps, start=1):
        step["n"] = i

    # Role lookup so the checklist can group/colour ingredients the same way the steps read.
    def role_of(token: str) -> str:
        if token == ings[0]:
            return "hero"
        if token in _AROMATICS:
            return "aromatic"
        if token in _TOMATO:
            return "base"
        if token in _RICH_DAIRY | _YOGURT | _CHEESE | _MILK:
            return "dairy"
        if token in _GARNISH:
            return "garnish"
        if token in _GREENS:
            return "veg"
        return "other"

    ingredients: list[dict[str, Any]] = []
    for token in ings:
        pantry = token in recommender.PANTRY
        ingredients.append({
            "token": token,
            "name": _name(token),
            "have": token in have,
            "pantry": pantry,
            "role": role_of(token),
            "substitutes": [] if pantry else _swap_ideas(
                token, own=own, exclude=exclude, candidates=candidates,
                k=sub_k, min_similarity=min_similarity,
            ),
        })

    have_count = sum(1 for i in ingredients if i["have"])
    return {
        "recipe_id": recipe.get("id"),
        "title": recipe.get("title"),
        "time_min": minutes,
        "tags": list(recipe.get("tags", [])),
        "method": method,
        "servings": servings,
        "note": "Steps are auto-generated from the recipe's ingredients and type — a helpful "
                "guide, not a fixed recipe. Adjust to taste.",
        "ingredients": ingredients,
        "have_count": have_count,
        "missing_count": len(ingredients) - have_count,
        "total_count": len(ingredients),
        "steps": steps,
        "total_timer_seconds": sum(s["seconds"] for s in steps),
    }
