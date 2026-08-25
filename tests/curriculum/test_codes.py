from app.curriculum.codes import (
    extract_filters_from_question,
    normalize_grade_code,
    normalize_subject_code,
)


def test_normalize_grade_variants():
    assert normalize_grade_code("Primary 4") == "CLASS_4"
    assert normalize_grade_code("CLASS_4") == "CLASS_4"
    assert normalize_grade_code("JSS 2") == "JSS_2"
    assert normalize_grade_code("SSS 1") == "SSS_1"


def test_normalize_subject():
    assert normalize_subject_code("Mathematics") == "MATHEMATICS"
    assert normalize_subject_code("maths") == "MATHEMATICS"
    assert normalize_subject_code("ENGLISH") == "ENGLISH"


def test_extract_filters_from_question():
    filters = extract_filters_from_question(
        "What topics are taught in Primary 4 Mathematics?"
    )
    assert filters["grade"] == "CLASS_4"
    assert filters["subject"] == "MATHEMATICS"
