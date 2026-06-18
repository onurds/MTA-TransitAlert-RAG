from __future__ import annotations

from pipeline.compiler.evidence import command_stripped_instruction, decompose_instruction


def test_evidence_decomposition_separates_control_alternative_and_temporal_units():
    instruction = (
        "Eastbound M66 buses will not stop on W 65th St at Columbus Ave. "
        "Use the stop on W 65th at Broadway instead. "
        "Timeframe is from today until Friday 10PM. "
        "Make sure to get it right."
    )

    units = decompose_instruction(instruction)

    typed_text = [(unit.unit_type, unit.text, unit.source) for unit in units]
    assert ("affected_service", "Eastbound M66 buses will not stop on W 65th St at Columbus Ave.", "instruction") in typed_text
    assert ("alternative_service", "Use the stop on W 65th at Broadway instead.", "instruction") in typed_text
    assert ("temporal_directive", "Timeframe is from today until Friday 10PM.", "instruction") in typed_text
    assert ("operator_control", "Make sure to get it right.", "instruction") in typed_text

    location_units = [unit for unit in units if unit.unit_type == "location_evidence"]
    assert any(unit.text == "W 65th St" and unit.source == "affected_service" for unit in location_units)
    assert any(unit.text == "Columbus Ave" and unit.source == "affected_service" for unit in location_units)
    assert any(unit.text == "W 65th" and unit.source == "alternative_service" for unit in location_units)
    assert any(unit.text == "Broadway instead" and unit.source == "alternative_service" for unit in location_units)


def test_command_stripped_instruction_keeps_rider_text_but_drops_operator_control():
    instruction = (
        "B20 buses are detoured at Decatur St at Wilson Ave. "
        "Use the stop on Decatur St at Irving Ave instead. "
        "Dates will be from now until 5 hours ahead. "
        "Make the dates match the timeframe above. "
        "Do not rewrite the route names. "
        "Do not change or rephrase the wording. "
        "Command: Abbreviate the timeframe on header a bit. "
        "Rule: Make sure to write the first sentence as is."
    )

    stripped = command_stripped_instruction(decompose_instruction(instruction))

    assert "B20 buses are detoured" in stripped
    assert "Use the stop on Decatur St at Irving Ave instead." in stripped
    assert "Dates will be" not in stripped
    assert "Make the dates match" not in stripped
    assert "Do not rewrite the route names" not in stripped
    assert "Do not change or rephrase" not in stripped
    assert "Abbreviate the timeframe" not in stripped
    assert "first sentence as is" not in stripped


def test_command_stripped_instruction_keeps_train_times_and_jamaica_place_names():
    instruction = (
        "The 8:57am train from Poughkeepsie to Grand Central is operating 10-15 minutes late. "
        "For service to/from bypassed stations, take a Jamaica or Forest Hills-bound train. "
        "Timeframe is tomorrow from 7 AM to 10 AM. "
        "Command: Abbreviate the timeframe on header a bit."
    )

    stripped = command_stripped_instruction(decompose_instruction(instruction))

    assert "8:57am train" in stripped
    assert "Jamaica or Forest Hills-bound train" in stripped
    assert "Timeframe is tomorrow" not in stripped
    assert "Abbreviate the timeframe" not in stripped
