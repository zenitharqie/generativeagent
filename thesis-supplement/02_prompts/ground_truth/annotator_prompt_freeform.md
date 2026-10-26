# Ground Truth Annotator Prompt — Free-form v3

You are one of three independent annotators constructing research ground truth for a
classroom multi-agent simulation study. Your judgments will be checked for inter-annotator
agreement against two other annotators who work separately from you.

There is NO predefined category list. You propose the categories yourself.

### Input you will receive

For each of 20 annotation units, in fixed canonical order:

- `canonical_persona`: the agent's identity, role, traits, goals, motivations.
- `canonical_event_description`: the target classroom event.
- `necessity_test`: the criteria a memory category must satisfy.

### Task

For every unit, propose the memory categories this agent genuinely NEEDS when responding
to that target event, based on the agent's persona and the event context.

### Proposing rules (mandatory)

1. Propose **3 to 5 categories per unit**. Fewer, precise categories are preferred.
2. Write each category as a **short descriptive phrase in English, 4 to 10 words**,
   stating a lasting behavioural tendency, need, or concern of the agent
   (example style: "Cautious preparation for upcoming written test").
3. Do not copy event wording verbatim; generalise to the underlying tendency,
   but keep it specific to this agent and event.
4. Every category must pass the Necessity Test: directly related to the agent's
   communication style, primary focus, or main motivation — AND contextually
   necessary for this event. Otherwise omit it.
5. Within one unit, do not propose two categories that mean the same thing.
6. Work independently. Do not infer other annotators' decisions or experimental outcomes.
7. The optional `reason` field (max 500 characters) must be a short, evidence-based
   justification citing the persona or event — no hidden chain-of-thought.

### Output format

Return ONLY a valid JSON array of exactly 20 objects, in the same canonical order as
the units were given, each shaped exactly like this (no additional keys):

```json
[
  {
    "annotation_unit_id": "<from the unit>",
    "agent_id": "<from the unit>",
    "target_event_id": "<from the unit>",
    "input_sha256": "<from the unit>",
    "categories": [
      {
        "category": "<your proposed short phrase>",
        "relevant": 1,
        "reason": "<optional, <=500 chars>"
      }
    ]
  }
]
```

Output raw JSON only — no markdown fences, no commentary before or after.
If the response would be cut off, continue the JSON array exactly where you stopped.

---

### Annotation units (canonical order — do not reorder)

```json
[
 {
  "annotation_unit_id": "gt_event_01_teacher",
  "agent_id": "teacher",
  "target_event_id": "event_01",
  "input_sha256": "bc1932d7a1a72838cfa8ee0b878fb4dffc86ec6a98742602f4ef720b4dfd8e15",
  "canonical_persona": {
   "agent_id": "teacher",
   "archetype": "calm_classroom_guide",
   "communication_style": "Clear, patient, concise, and instructional.",
   "display_name": "Ms. Sunday",
   "goals": [
    "help students understand classroom material",
    "maintain fair and clear classroom expectations",
    "support balanced student participation"
   ],
   "motivations": [
    "student understanding",
    "consistent classroom norms",
    "constructive participation"
   ],
   "persona_summary": "A calm and structured teacher who gives clear guidance, applies classroom expectations consistently, and invites students to participate respectfully.",
   "role": "teacher",
   "traits": [
    "calm",
    "structured",
    "attentive"
   ]
  },
  "canonical_event_description": "Ms. Sunday announces that a written test will take place on Day 6 and asks the class to prepare using the material discussed during the week.",
  "necessity_test": {
   "context_requirement": "the category must also be relevant to the target-event context",
   "criteria": [
    "directly related to the communication style defined in the agent Base Prompt",
    "directly related to the agent's primary focus or topic of attention",
    "directly related to the agent's primary motivation or goal"
   ],
   "decision": "binary_relevance",
   "not_relevant_value": 0,
   "relevant_value": 1
  }
 },
 {
  "annotation_unit_id": "gt_event_01_active_student",
  "agent_id": "active_student",
  "target_event_id": "event_01",
  "input_sha256": "90466e2a7f1ce8bed0f4643c1f612b3658f420e673481e1a3fe1ddd287e03c5d",
  "canonical_persona": {
   "agent_id": "active_student",
   "archetype": "active_social_learner",
   "communication_style": "Friendly, direct, energetic, and willing to speak first.",
   "display_name": "Monday",
   "goals": [
    "understand lessons by discussing them",
    "encourage classmates to participate",
    "clarify uncertainty by asking questions"
   ],
   "motivations": [
    "social learning",
    "shared understanding",
    "active participation"
   ],
   "persona_summary": "An outgoing and curious student who learns through questions, discussion, and active participation with classmates.",
   "role": "student",
   "traits": [
    "curious",
    "outgoing",
    "collaborative"
   ]
  },
  "canonical_event_description": "Ms. Sunday announces that a written test will take place on Day 6 and asks the class to prepare using the material discussed during the week.",
  "necessity_test": {
   "context_requirement": "the category must also be relevant to the target-event context",
   "criteria": [
    "directly related to the communication style defined in the agent Base Prompt",
    "directly related to the agent's primary focus or topic of attention",
    "directly related to the agent's primary motivation or goal"
   ],
   "decision": "binary_relevance",
   "not_relevant_value": 0,
   "relevant_value": 1
  }
 },
 {
  "annotation_unit_id": "gt_event_01_quiet_student",
  "agent_id": "quiet_student",
  "target_event_id": "event_01",
  "input_sha256": "10642f479d77bd7c2644436d37d4abbbf0643d5a98ecdeedc15c578849d083c7",
  "canonical_persona": {
   "agent_id": "quiet_student",
   "archetype": "quiet_careful_observer",
   "communication_style": "Brief, cautious, respectful, and deliberate.",
   "display_name": "Tuesday",
   "goals": [
    "understand classroom expectations",
    "participate without unnecessary attention",
    "prepare a clear response before speaking"
   ],
   "motivations": [
    "clarity",
    "preparedness",
    "a respectful classroom atmosphere"
   ],
   "persona_summary": "A reserved and observant student who listens carefully, prepares before speaking, and prefers concise responses.",
   "role": "student",
   "traits": [
    "reserved",
    "careful",
    "observant"
   ]
  },
  "canonical_event_description": "Ms. Sunday announces that a written test will take place on Day 6 and asks the class to prepare using the material discussed during the week.",
  "necessity_test": {
   "context_requirement": "the category must also be relevant to the target-event context",
   "criteria": [
    "directly related to the communication style defined in the agent Base Prompt",
    "directly related to the agent's primary focus or topic of attention",
    "directly related to the agent's primary motivation or goal"
   ],
   "decision": "binary_relevance",
   "not_relevant_value": 0,
   "relevant_value": 1
  }
 },
 {
  "annotation_unit_id": "gt_event_01_academic_student",
  "agent_id": "academic_student",
  "target_event_id": "event_01",
  "input_sha256": "acf92dbe85ee5bf4031682cd2b19e62e07f0221bb5694417f0a3d0fd9c6c2812",
  "canonical_persona": {
   "agent_id": "academic_student",
   "archetype": "structured_academic_learner",
   "communication_style": "Precise, composed, academically focused, and structured.",
   "display_name": "Wednesday",
   "goals": [
    "master the classroom material",
    "maintain an organized study routine",
    "contribute accurate details during class work"
   ],
   "motivations": [
    "subject mastery",
    "academic preparation",
    "accurate understanding"
   ],
   "persona_summary": "A focused and achievement-oriented student who organizes material carefully and prefers evidence-based study strategies.",
   "role": "student",
   "traits": [
    "focused",
    "organized",
    "analytical"
   ]
  },
  "canonical_event_description": "Ms. Sunday announces that a written test will take place on Day 6 and asks the class to prepare using the material discussed during the week.",
  "necessity_test": {
   "context_requirement": "the category must also be relevant to the target-event context",
   "criteria": [
    "directly related to the communication style defined in the agent Base Prompt",
    "directly related to the agent's primary focus or topic of attention",
    "directly related to the agent's primary motivation or goal"
   ],
   "decision": "binary_relevance",
   "not_relevant_value": 0,
   "relevant_value": 1
  }
 },
 {
  "annotation_unit_id": "gt_event_02_teacher",
  "agent_id": "teacher",
  "target_event_id": "event_02",
  "input_sha256": "3c64a383ef5577c8d6b1e23a24ca7ef10e90279070ef25d881c70cad4c30b3f3",
  "canonical_persona": {
   "agent_id": "teacher",
   "archetype": "calm_classroom_guide",
   "communication_style": "Clear, patient, concise, and instructional.",
   "display_name": "Ms. Sunday",
   "goals": [
    "help students understand classroom material",
    "maintain fair and clear classroom expectations",
    "support balanced student participation"
   ],
   "motivations": [
    "student understanding",
    "consistent classroom norms",
    "constructive participation"
   ],
   "persona_summary": "A calm and structured teacher who gives clear guidance, applies classroom expectations consistently, and invites students to participate respectfully.",
   "role": "teacher",
   "traits": [
    "calm",
    "structured",
    "attentive"
   ]
  },
  "canonical_event_description": "Ms. Sunday announces that students must raise a hand and wait to be acknowledged before speaking during class discussion.",
  "necessity_test": {
   "context_requirement": "the category must also be relevant to the target-event context",
   "criteria": [
    "directly related to the communication style defined in the agent Base Prompt",
    "directly related to the agent's primary focus or topic of attention",
    "directly related to the agent's primary motivation or goal"
   ],
   "decision": "binary_relevance",
   "not_relevant_value": 0,
   "relevant_value": 1
  }
 },
 {
  "annotation_unit_id": "gt_event_02_active_student",
  "agent_id": "active_student",
  "target_event_id": "event_02",
  "input_sha256": "599672c47bca5bfc26edf335bf87b881006db074de82bf0b513615dfe2216c66",
  "canonical_persona": {
   "agent_id": "active_student",
   "archetype": "active_social_learner",
   "communication_style": "Friendly, direct, energetic, and willing to speak first.",
   "display_name": "Monday",
   "goals": [
    "understand lessons by discussing them",
    "encourage classmates to participate",
    "clarify uncertainty by asking questions"
   ],
   "motivations": [
    "social learning",
    "shared understanding",
    "active participation"
   ],
   "persona_summary": "An outgoing and curious student who learns through questions, discussion, and active participation with classmates.",
   "role": "student",
   "traits": [
    "curious",
    "outgoing",
    "collaborative"
   ]
  },
  "canonical_event_description": "Ms. Sunday announces that students must raise a hand and wait to be acknowledged before speaking during class discussion.",
  "necessity_test": {
   "context_requirement": "the category must also be relevant to the target-event context",
   "criteria": [
    "directly related to the communication style defined in the agent Base Prompt",
    "directly related to the agent's primary focus or topic of attention",
    "directly related to the agent's primary motivation or goal"
   ],
   "decision": "binary_relevance",
   "not_relevant_value": 0,
   "relevant_value": 1
  }
 },
 {
  "annotation_unit_id": "gt_event_02_quiet_student",
  "agent_id": "quiet_student",
  "target_event_id": "event_02",
  "input_sha256": "3685d4467a5974ff3e5656b17f767ec29c015f0528a45303eaf2f63e1a5fb7e2",
  "canonical_persona": {
   "agent_id": "quiet_student",
   "archetype": "quiet_careful_observer",
   "communication_style": "Brief, cautious, respectful, and deliberate.",
   "display_name": "Tuesday",
   "goals": [
    "understand classroom expectations",
    "participate without unnecessary attention",
    "prepare a clear response before speaking"
   ],
   "motivations": [
    "clarity",
    "preparedness",
    "a respectful classroom atmosphere"
   ],
   "persona_summary": "A reserved and observant student who listens carefully, prepares before speaking, and prefers concise responses.",
   "role": "student",
   "traits": [
    "reserved",
    "careful",
    "observant"
   ]
  },
  "canonical_event_description": "Ms. Sunday announces that students must raise a hand and wait to be acknowledged before speaking during class discussion.",
  "necessity_test": {
   "context_requirement": "the category must also be relevant to the target-event context",
   "criteria": [
    "directly related to the communication style defined in the agent Base Prompt",
    "directly related to the agent's primary focus or topic of attention",
    "directly related to the agent's primary motivation or goal"
   ],
   "decision": "binary_relevance",
   "not_relevant_value": 0,
   "relevant_value": 1
  }
 },
 {
  "annotation_unit_id": "gt_event_02_academic_student",
  "agent_id": "academic_student",
  "target_event_id": "event_02",
  "input_sha256": "b2eca2dfba47ed688954db09347766ba16a91476961cb8776eb07f22461574e8",
  "canonical_persona": {
   "agent_id": "academic_student",
   "archetype": "structured_academic_learner",
   "communication_style": "Precise, composed, academically focused, and structured.",
   "display_name": "Wednesday",
   "goals": [
    "master the classroom material",
    "maintain an organized study routine",
    "contribute accurate details during class work"
   ],
   "motivations": [
    "subject mastery",
    "academic preparation",
    "accurate understanding"
   ],
   "persona_summary": "A focused and achievement-oriented student who organizes material carefully and prefers evidence-based study strategies.",
   "role": "student",
   "traits": [
    "focused",
    "organized",
    "analytical"
   ]
  },
  "canonical_event_description": "Ms. Sunday announces that students must raise a hand and wait to be acknowledged before speaking during class discussion.",
  "necessity_test": {
   "context_requirement": "the category must also be relevant to the target-event context",
   "criteria": [
    "directly related to the communication style defined in the agent Base Prompt",
    "directly related to the agent's primary focus or topic of attention",
    "directly related to the agent's primary motivation or goal"
   ],
   "decision": "binary_relevance",
   "not_relevant_value": 0,
   "relevant_value": 1
  }
 },
 {
  "annotation_unit_id": "gt_event_03_teacher",
  "agent_id": "teacher",
  "target_event_id": "event_03",
  "input_sha256": "6eb3abd1fe77a7bda92a022d81cbd8ea8528fbcc6feb158efc1bafbb93cd6f66",
  "canonical_persona": {
   "agent_id": "teacher",
   "archetype": "calm_classroom_guide",
   "communication_style": "Clear, patient, concise, and instructional.",
   "display_name": "Ms. Sunday",
   "goals": [
    "help students understand classroom material",
    "maintain fair and clear classroom expectations",
    "support balanced student participation"
   ],
   "motivations": [
    "student understanding",
    "consistent classroom norms",
    "constructive participation"
   ],
   "persona_summary": "A calm and structured teacher who gives clear guidance, applies classroom expectations consistently, and invites students to participate respectfully.",
   "role": "teacher",
   "traits": [
    "calm",
    "structured",
    "attentive"
   ]
  },
  "canonical_event_description": "Ms. Sunday assigns the students a group discussion and asks each student to contribute to a shared explanation of the current lesson.",
  "necessity_test": {
   "context_requirement": "the category must also be relevant to the target-event context",
   "criteria": [
    "directly related to the communication style defined in the agent Base Prompt",
    "directly related to the agent's primary focus or topic of attention",
    "directly related to the agent's primary motivation or goal"
   ],
   "decision": "binary_relevance",
   "not_relevant_value": 0,
   "relevant_value": 1
  }
 },
 {
  "annotation_unit_id": "gt_event_03_active_student",
  "agent_id": "active_student",
  "target_event_id": "event_03",
  "input_sha256": "6728644ca832b7cc985158022934edf507fe4529794af618d170a23026a39c7f",
  "canonical_persona": {
   "agent_id": "active_student",
   "archetype": "active_social_learner",
   "communication_style": "Friendly, direct, energetic, and willing to speak first.",
   "display_name": "Monday",
   "goals": [
    "understand lessons by discussing them",
    "encourage classmates to participate",
    "clarify uncertainty by asking questions"
   ],
   "motivations": [
    "social learning",
    "shared understanding",
    "active participation"
   ],
   "persona_summary": "An outgoing and curious student who learns through questions, discussion, and active participation with classmates.",
   "role": "student",
   "traits": [
    "curious",
    "outgoing",
    "collaborative"
   ]
  },
  "canonical_event_description": "Ms. Sunday assigns the students a group discussion and asks each student to contribute to a shared explanation of the current lesson.",
  "necessity_test": {
   "context_requirement": "the category must also be relevant to the target-event context",
   "criteria": [
    "directly related to the communication style defined in the agent Base Prompt",
    "directly related to the agent's primary focus or topic of attention",
    "directly related to the agent's primary motivation or goal"
   ],
   "decision": "binary_relevance",
   "not_relevant_value": 0,
   "relevant_value": 1
  }
 },
 {
  "annotation_unit_id": "gt_event_03_quiet_student",
  "agent_id": "quiet_student",
  "target_event_id": "event_03",
  "input_sha256": "119f70bdb51fc7a308f963fc618cf28312ed5edd8530522db439f2f20dbd4977",
  "canonical_persona": {
   "agent_id": "quiet_student",
   "archetype": "quiet_careful_observer",
   "communication_style": "Brief, cautious, respectful, and deliberate.",
   "display_name": "Tuesday",
   "goals": [
    "understand classroom expectations",
    "participate without unnecessary attention",
    "prepare a clear response before speaking"
   ],
   "motivations": [
    "clarity",
    "preparedness",
    "a respectful classroom atmosphere"
   ],
   "persona_summary": "A reserved and observant student who listens carefully, prepares before speaking, and prefers concise responses.",
   "role": "student",
   "traits": [
    "reserved",
    "careful",
    "observant"
   ]
  },
  "canonical_event_description": "Ms. Sunday assigns the students a group discussion and asks each student to contribute to a shared explanation of the current lesson.",
  "necessity_test": {
   "context_requirement": "the category must also be relevant to the target-event context",
   "criteria": [
    "directly related to the communication style defined in the agent Base Prompt",
    "directly related to the agent's primary focus or topic of attention",
    "directly related to the agent's primary motivation or goal"
   ],
   "decision": "binary_relevance",
   "not_relevant_value": 0,
   "relevant_value": 1
  }
 },
 {
  "annotation_unit_id": "gt_event_03_academic_student",
  "agent_id": "academic_student",
  "target_event_id": "event_03",
  "input_sha256": "587d602fd6aab3e32a0b6295785f4001338c374eae2a5027ce6ec91a5467fa77",
  "canonical_persona": {
   "agent_id": "academic_student",
   "archetype": "structured_academic_learner",
   "communication_style": "Precise, composed, academically focused, and structured.",
   "display_name": "Wednesday",
   "goals": [
    "master the classroom material",
    "maintain an organized study routine",
    "contribute accurate details during class work"
   ],
   "motivations": [
    "subject mastery",
    "academic preparation",
    "accurate understanding"
   ],
   "persona_summary": "A focused and achievement-oriented student who organizes material carefully and prefers evidence-based study strategies.",
   "role": "student",
   "traits": [
    "focused",
    "organized",
    "analytical"
   ]
  },
  "canonical_event_description": "Ms. Sunday assigns the students a group discussion and asks each student to contribute to a shared explanation of the current lesson.",
  "necessity_test": {
   "context_requirement": "the category must also be relevant to the target-event context",
   "criteria": [
    "directly related to the communication style defined in the agent Base Prompt",
    "directly related to the agent's primary focus or topic of attention",
    "directly related to the agent's primary motivation or goal"
   ],
   "decision": "binary_relevance",
   "not_relevant_value": 0,
   "relevant_value": 1
  }
 },
 {
  "annotation_unit_id": "gt_event_04_teacher",
  "agent_id": "teacher",
  "target_event_id": "event_04",
  "input_sha256": "084f025ef2cee11b963df650476e3fd00d5e7f0ceb456ce5599dcca04599c263",
  "canonical_persona": {
   "agent_id": "teacher",
   "archetype": "calm_classroom_guide",
   "communication_style": "Clear, patient, concise, and instructional.",
   "display_name": "Ms. Sunday",
   "goals": [
    "help students understand classroom material",
    "maintain fair and clear classroom expectations",
    "support balanced student participation"
   ],
   "motivations": [
    "student understanding",
    "consistent classroom norms",
    "constructive participation"
   ],
   "persona_summary": "A calm and structured teacher who gives clear guidance, applies classroom expectations consistently, and invites students to participate respectfully.",
   "role": "teacher",
   "traits": [
    "calm",
    "structured",
    "attentive"
   ]
  },
  "canonical_event_description": "Ms. Sunday asks the class to nominate and discuss a student who could take responsibility as class leader.",
  "necessity_test": {
   "context_requirement": "the category must also be relevant to the target-event context",
   "criteria": [
    "directly related to the communication style defined in the agent Base Prompt",
    "directly related to the agent's primary focus or topic of attention",
    "directly related to the agent's primary motivation or goal"
   ],
   "decision": "binary_relevance",
   "not_relevant_value": 0,
   "relevant_value": 1
  }
 },
 {
  "annotation_unit_id": "gt_event_04_active_student",
  "agent_id": "active_student",
  "target_event_id": "event_04",
  "input_sha256": "25ce9299bf2df4e0095277d8faebc1732e2edd7bf4e66ad44f4847c5ce7942f4",
  "canonical_persona": {
   "agent_id": "active_student",
   "archetype": "active_social_learner",
   "communication_style": "Friendly, direct, energetic, and willing to speak first.",
   "display_name": "Monday",
   "goals": [
    "understand lessons by discussing them",
    "encourage classmates to participate",
    "clarify uncertainty by asking questions"
   ],
   "motivations": [
    "social learning",
    "shared understanding",
    "active participation"
   ],
   "persona_summary": "An outgoing and curious student who learns through questions, discussion, and active participation with classmates.",
   "role": "student",
   "traits": [
    "curious",
    "outgoing",
    "collaborative"
   ]
  },
  "canonical_event_description": "Ms. Sunday asks the class to nominate and discuss a student who could take responsibility as class leader.",
  "necessity_test": {
   "context_requirement": "the category must also be relevant to the target-event context",
   "criteria": [
    "directly related to the communication style defined in the agent Base Prompt",
    "directly related to the agent's primary focus or topic of attention",
    "directly related to the agent's primary motivation or goal"
   ],
   "decision": "binary_relevance",
   "not_relevant_value": 0,
   "relevant_value": 1
  }
 },
 {
  "annotation_unit_id": "gt_event_04_quiet_student",
  "agent_id": "quiet_student",
  "target_event_id": "event_04",
  "input_sha256": "b85fadfed17cebf43bc1bb536f8905137c4ccaf704bf36490827e0479da3e6b8",
  "canonical_persona": {
   "agent_id": "quiet_student",
   "archetype": "quiet_careful_observer",
   "communication_style": "Brief, cautious, respectful, and deliberate.",
   "display_name": "Tuesday",
   "goals": [
    "understand classroom expectations",
    "participate without unnecessary attention",
    "prepare a clear response before speaking"
   ],
   "motivations": [
    "clarity",
    "preparedness",
    "a respectful classroom atmosphere"
   ],
   "persona_summary": "A reserved and observant student who listens carefully, prepares before speaking, and prefers concise responses.",
   "role": "student",
   "traits": [
    "reserved",
    "careful",
    "observant"
   ]
  },
  "canonical_event_description": "Ms. Sunday asks the class to nominate and discuss a student who could take responsibility as class leader.",
  "necessity_test": {
   "context_requirement": "the category must also be relevant to the target-event context",
   "criteria": [
    "directly related to the communication style defined in the agent Base Prompt",
    "directly related to the agent's primary focus or topic of attention",
    "directly related to the agent's primary motivation or goal"
   ],
   "decision": "binary_relevance",
   "not_relevant_value": 0,
   "relevant_value": 1
  }
 },
 {
  "annotation_unit_id": "gt_event_04_academic_student",
  "agent_id": "academic_student",
  "target_event_id": "event_04",
  "input_sha256": "02dff8c7090d3c6c8fe1934185da272b8021de0fb6ce27089a2a8023f3726911",
  "canonical_persona": {
   "agent_id": "academic_student",
   "archetype": "structured_academic_learner",
   "communication_style": "Precise, composed, academically focused, and structured.",
   "display_name": "Wednesday",
   "goals": [
    "master the classroom material",
    "maintain an organized study routine",
    "contribute accurate details during class work"
   ],
   "motivations": [
    "subject mastery",
    "academic preparation",
    "accurate understanding"
   ],
   "persona_summary": "A focused and achievement-oriented student who organizes material carefully and prefers evidence-based study strategies.",
   "role": "student",
   "traits": [
    "focused",
    "organized",
    "analytical"
   ]
  },
  "canonical_event_description": "Ms. Sunday asks the class to nominate and discuss a student who could take responsibility as class leader.",
  "necessity_test": {
   "context_requirement": "the category must also be relevant to the target-event context",
   "criteria": [
    "directly related to the communication style defined in the agent Base Prompt",
    "directly related to the agent's primary focus or topic of attention",
    "directly related to the agent's primary motivation or goal"
   ],
   "decision": "binary_relevance",
   "not_relevant_value": 0,
   "relevant_value": 1
  }
 },
 {
  "annotation_unit_id": "gt_event_05_teacher",
  "agent_id": "teacher",
  "target_event_id": "event_05",
  "input_sha256": "8a582dd344f404d51c1af459c4c812a1c585aff3ea715194f77d5f62925365fc",
  "canonical_persona": {
   "agent_id": "teacher",
   "archetype": "calm_classroom_guide",
   "communication_style": "Clear, patient, concise, and instructional.",
   "display_name": "Ms. Sunday",
   "goals": [
    "help students understand classroom material",
    "maintain fair and clear classroom expectations",
    "support balanced student participation"
   ],
   "motivations": [
    "student understanding",
    "consistent classroom norms",
    "constructive participation"
   ],
   "persona_summary": "A calm and structured teacher who gives clear guidance, applies classroom expectations consistently, and invites students to participate respectfully.",
   "role": "teacher",
   "traits": [
    "calm",
    "structured",
    "attentive"
   ]
  },
  "canonical_event_description": "Ms. Sunday begins the announced written test and asks each student to respond based on their preparation and understanding.",
  "necessity_test": {
   "context_requirement": "the category must also be relevant to the target-event context",
   "criteria": [
    "directly related to the communication style defined in the agent Base Prompt",
    "directly related to the agent's primary focus or topic of attention",
    "directly related to the agent's primary motivation or goal"
   ],
   "decision": "binary_relevance",
   "not_relevant_value": 0,
   "relevant_value": 1
  }
 },
 {
  "annotation_unit_id": "gt_event_05_active_student",
  "agent_id": "active_student",
  "target_event_id": "event_05",
  "input_sha256": "212af19e2c00318d92b07b9fe5426be614540e67bd6c4806beeba7555e4ff145",
  "canonical_persona": {
   "agent_id": "active_student",
   "archetype": "active_social_learner",
   "communication_style": "Friendly, direct, energetic, and willing to speak first.",
   "display_name": "Monday",
   "goals": [
    "understand lessons by discussing them",
    "encourage classmates to participate",
    "clarify uncertainty by asking questions"
   ],
   "motivations": [
    "social learning",
    "shared understanding",
    "active participation"
   ],
   "persona_summary": "An outgoing and curious student who learns through questions, discussion, and active participation with classmates.",
   "role": "student",
   "traits": [
    "curious",
    "outgoing",
    "collaborative"
   ]
  },
  "canonical_event_description": "Ms. Sunday begins the announced written test and asks each student to respond based on their preparation and understanding.",
  "necessity_test": {
   "context_requirement": "the category must also be relevant to the target-event context",
   "criteria": [
    "directly related to the communication style defined in the agent Base Prompt",
    "directly related to the agent's primary focus or topic of attention",
    "directly related to the agent's primary motivation or goal"
   ],
   "decision": "binary_relevance",
   "not_relevant_value": 0,
   "relevant_value": 1
  }
 },
 {
  "annotation_unit_id": "gt_event_05_quiet_student",
  "agent_id": "quiet_student",
  "target_event_id": "event_05",
  "input_sha256": "f63388907186508fcb1073b9dd74f74835a074095ca29b68dd70b1153868ff3d",
  "canonical_persona": {
   "agent_id": "quiet_student",
   "archetype": "quiet_careful_observer",
   "communication_style": "Brief, cautious, respectful, and deliberate.",
   "display_name": "Tuesday",
   "goals": [
    "understand classroom expectations",
    "participate without unnecessary attention",
    "prepare a clear response before speaking"
   ],
   "motivations": [
    "clarity",
    "preparedness",
    "a respectful classroom atmosphere"
   ],
   "persona_summary": "A reserved and observant student who listens carefully, prepares before speaking, and prefers concise responses.",
   "role": "student",
   "traits": [
    "reserved",
    "careful",
    "observant"
   ]
  },
  "canonical_event_description": "Ms. Sunday begins the announced written test and asks each student to respond based on their preparation and understanding.",
  "necessity_test": {
   "context_requirement": "the category must also be relevant to the target-event context",
   "criteria": [
    "directly related to the communication style defined in the agent Base Prompt",
    "directly related to the agent's primary focus or topic of attention",
    "directly related to the agent's primary motivation or goal"
   ],
   "decision": "binary_relevance",
   "not_relevant_value": 0,
   "relevant_value": 1
  }
 },
 {
  "annotation_unit_id": "gt_event_05_academic_student",
  "agent_id": "academic_student",
  "target_event_id": "event_05",
  "input_sha256": "716ad6704a73e86b75d761d4c7dfb6d364ed5599152a446c1b4c943541907174",
  "canonical_persona": {
   "agent_id": "academic_student",
   "archetype": "structured_academic_learner",
   "communication_style": "Precise, composed, academically focused, and structured.",
   "display_name": "Wednesday",
   "goals": [
    "master the classroom material",
    "maintain an organized study routine",
    "contribute accurate details during class work"
   ],
   "motivations": [
    "subject mastery",
    "academic preparation",
    "accurate understanding"
   ],
   "persona_summary": "A focused and achievement-oriented student who organizes material carefully and prefers evidence-based study strategies.",
   "role": "student",
   "traits": [
    "focused",
    "organized",
    "analytical"
   ]
  },
  "canonical_event_description": "Ms. Sunday begins the announced written test and asks each student to respond based on their preparation and understanding.",
  "necessity_test": {
   "context_requirement": "the category must also be relevant to the target-event context",
   "criteria": [
    "directly related to the communication style defined in the agent Base Prompt",
    "directly related to the agent's primary focus or topic of attention",
    "directly related to the agent's primary motivation or goal"
   ],
   "decision": "binary_relevance",
   "not_relevant_value": 0,
   "relevant_value": 1
  }
 }
]

---

End of input.
