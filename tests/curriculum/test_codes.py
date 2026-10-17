from app.curriculum.codes import (
    extract_filters_from_question,
    normalize_classification,
    normalize_grade_code,
    normalize_subject_code,
)


def test_normalize_classification_variants():
    assert normalize_classification("core") == "CORE"
    assert normalize_classification("non-core") == "NON_CORE"
    assert normalize_classification("NON_CORE") == "NON_CORE"
    assert normalize_classification("optional") == "OPTIONAL"


def test_normalize_grade_variants():
    assert normalize_grade_code("Primary 4") == "CLASS_4"
    assert normalize_grade_code("CLASS_4") == "CLASS_4"
    assert normalize_grade_code("JSS 2") == "JSS_2"
    assert normalize_grade_code("SSS 1") == "SSS_1"
    assert normalize_grade_code("SSS1") == "SSS_1"
    assert normalize_grade_code("SSS-1") == "SSS_1"
    assert normalize_grade_code("SSS_2") == "SSS_2"
    assert normalize_grade_code("Senior Secondary 3") == "SSS_3"
    assert normalize_grade_code("Senior Secondary School 1") == "SSS_1"


def test_normalize_subject():
    assert normalize_subject_code("Mathematics") == "MATHEMATICS"
    assert normalize_subject_code("maths") == "MATHEMATICS"
    assert normalize_subject_code("ENGLISH") == "ENGLISH"
    assert normalize_subject_code("English") == "ENGLISH"
    assert normalize_subject_code("Science") == "SCIENCE"
    assert normalize_subject_code("Biology") == "BIOLOGY"
    assert normalize_subject_code("English Language") == "ENGLISH_LANGUAGE"
    assert normalize_subject_code("Integrated Science") == "INTEGRATED_SCIENCE"
    assert normalize_subject_code("English Science") is None


def test_extract_filters_from_question():
    filters = extract_filters_from_question(
        "What topics are taught in Primary 4 Mathematics?"
    )
    assert filters["grade"] == "CLASS_4"
    assert filters["subject"] == "MATHEMATICS"
    assert filters["classification"] is None


def test_gold_french_question_extracts_french():
    filters = extract_filters_from_question(
        "What units does the Class 4 French subject cover?"
    )
    assert filters["grade"] == "CLASS_4"
    assert filters["subject"] == "FRENCH"


def test_french_appreciation_is_not_shortened_to_french():
    filters = extract_filters_from_question(
        "What units does Class 4 French Appreciation cover?"
    )
    assert filters["grade"] == "CLASS_4"
    assert filters["subject"] == "FRENCH_APPRECIATION"


def test_sss_french_as_a_foreign_language_is_not_shortened_to_french():
    filters = extract_filters_from_question(
        "What topics are in SSS 1 French as a Foreign Language?"
    )
    assert filters["grade"] == "SSS_1"
    assert filters["subject"] == "FRENCH_FOREIGN_LANGUAGE"


def test_french_subject_names_normalize_to_published_codes():
    assert normalize_subject_code("French") == "FRENCH"
    assert normalize_subject_code("French Appreciation") == "FRENCH_APPRECIATION"
    assert (
        normalize_subject_code("French as a Foreign Language")
        == "FRENCH_FOREIGN_LANGUAGE"
    )
    assert normalize_subject_code("FRENCH_APPRECIATION") == "FRENCH_APPRECIATION"


def test_existing_subject_distinctions_are_unchanged():
    cases = {
        "What units does Class 4 Science cover?": ("CLASS_4", "SCIENCE"),
        "What is the first unit of JSS 1 Integrated Science?": (
            "JSS_1",
            "INTEGRATED_SCIENCE",
        ),
        "What units does Class 4 Agricultural Science cover?": (
            "CLASS_4",
            "AGRICULTURAL_SCIENCE",
        ),
        "What units are in Class 4 English Language?": ("CLASS_4", "ENGLISH_LANGUAGE"),
        "List the JSS 2 Mathematics units.": ("JSS_2", "MATHEMATICS"),
        "What topics are in SSS 1 Biology?": ("SSS_1", None),
    }
    for question, (grade, subject) in cases.items():
        filters = extract_filters_from_question(question)
        assert (filters["grade"], filters["subject"]) == (grade, subject), question


def test_extract_core_classification_filter():
    filters = extract_filters_from_question(
        "What are the core subjects in Primary 3?"
    )
    assert filters["grade"] == "CLASS_3"
    assert filters["classification"] == "CORE"
