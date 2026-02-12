"""
Professional job opportunity postscript messages for LinkedIn posts.

These messages convey openness to new opportunities in a non-begging,
professional, and engaging manner suitable for any professional discipline.
"""

import random
from config import Config


def _get_discipline_field() -> str:
    """Get the discipline as a field (e.g., 'chemical engineering' from 'chemical engineer')."""
    discipline = Config.DISCIPLINE
    # Convert 'engineer' to 'engineering' for field context
    return discipline.replace(" engineer", " engineering").replace(
        " Engineer", " Engineering"
    )


# 50 professional postscript messages about job opportunities
# Messages use {discipline} and {discipline_field} placeholders
OPPORTUNITY_MESSAGE_TEMPLATES = [
    # Direct but professional
    "P.S. I'm currently exploring new opportunities in {discipline_field}. Let's connect!",
    "P.S. Open to discussing exciting {discipline_field} roles. Feel free to reach out!",
    "P.S. I'm actively seeking my next challenge in the {discipline_field} space.",
    "P.S. Looking for my next opportunity to make an impact in {discipline_field}.",
    "P.S. Currently exploring new chapters in my {discipline_field} career.",
    # Expertise-focused
    "P.S. I bring process optimization expertise to every role. Open to new opportunities!",
    "P.S. Seeking roles where I can apply my {discipline_field} skills to real-world challenges.",
    "P.S. Looking to bring my technical expertise to an innovative team. Let's talk!",
    "P.S. Ready to contribute my engineering experience to a forward-thinking organization.",
    "P.S. Eager to apply my {discipline_field} background to new challenges.",
    # Value proposition
    "P.S. If your team needs someone who can bridge technical and business needs, I'd love to chat.",
    "P.S. I help organizations optimize their processes. Currently available for new roles.",
    "P.S. Looking to join a team where I can drive efficiency and innovation.",
    "P.S. Seeking opportunities to leverage my expertise in process improvement.",
    "P.S. Open to roles where I can make a measurable difference.",
    # Networking-oriented
    "P.S. Always happy to connect with fellow professionals. Currently open to opportunities!",
    "P.S. My network has been invaluable. If you know of any openings, I'd appreciate a referral!",
    "P.S. Building connections while exploring new career opportunities. Let's connect!",
    "P.S. Your network might hold my next opportunity. Happy to connect!",
    "P.S. Expanding my professional network while seeking new challenges.",
    # Industry-specific
    "P.S. Passionate about sustainable processes. Open to opportunities in this space!",
    "P.S. Seeking roles in process safety, optimization, or operations.",
    "P.S. Looking for opportunities in industry sectors where I can add value.",
    "P.S. Open to roles in R&D, process development, or manufacturing excellence.",
    "P.S. Interested in opportunities at the intersection of engineering and innovation.",
    # Soft approach
    "P.S. If this content resonates, perhaps we should connect. I'm exploring new paths.",
    "P.S. Enjoying sharing industry insights while I search for my next role.",
    "P.S. These posts keep me connected to the industry I love. Open to new opportunities!",
    "P.S. Staying engaged with the community while exploring career transitions.",
    "P.S. Sharing knowledge and seeking new professional adventures.",
    # Conversation starters
    "P.S. I'd love to hear about interesting challenges you're facing. Also open to new roles!",
    "P.S. What's the most exciting problem you're solving? I'm looking to tackle new ones myself.",
    "P.S. Always up for a conversation about {discipline_field}. And yes, I'm job hunting!",
    "P.S. Let's discuss the future of {discipline_field}. I'm also exploring new opportunities.",
    "P.S. Coffee chat? I'm always learning and currently looking for my next role.",
    # Confidence-building
    "P.S. My track record speaks for itself. Ready for new challenges!",
    "P.S. Proven results in process optimization. Currently on the market.",
    "P.S. I've helped teams achieve breakthrough results. Looking for my next team.",
    "P.S. A decade of engineering excellence, seeking a new home.",
    "P.S. My experience could be your competitive advantage. Let's talk!",
    # Collaborative tone
    "P.S. Looking to join a team that values innovation and collaboration.",
    "P.S. Seeking a role where I can both contribute and grow.",
    "P.S. Open to opportunities where mentorship goes both ways.",
    "P.S. Looking for a team that tackles complex challenges together.",
    "P.S. Seeking a collaborative environment for my next career chapter.",
    # Forward-looking
    "P.S. Excited about where {discipline_field} is heading. Looking to be part of it!",
    "P.S. The industry is evolving rapidly. I want to help shape its future.",
    "P.S. Seeking roles at the cutting edge of {discipline_field}.",
    "P.S. Ready to contribute to the next generation of process innovation.",
    "P.S. Looking for opportunities to work on tomorrow's engineering challenges.",
]


def _format_message(template: str) -> str:
    """Format a message template with the current discipline."""
    return template.format(
        discipline=Config.DISCIPLINE, discipline_field=_get_discipline_field()
    )


def get_random_opportunity_message() -> str:
    """Get a randomly selected opportunity message."""
    template = random.choice(OPPORTUNITY_MESSAGE_TEMPLATES)
    return _format_message(template)


def get_opportunity_message_by_index(index: int) -> str:
    """Get a specific opportunity message by index (0-49)."""
    template = OPPORTUNITY_MESSAGE_TEMPLATES[index % len(OPPORTUNITY_MESSAGE_TEMPLATES)]
    return _format_message(template)


def get_all_messages() -> list[str]:
    """Get all opportunity messages."""
    return [_format_message(t) for t in OPPORTUNITY_MESSAGE_TEMPLATES]


def get_message_count() -> int:
    """Get the total number of available messages."""
    return len(OPPORTUNITY_MESSAGE_TEMPLATES)


# Optional: Allow customization via config
_custom_messages: list[str] = []


def add_custom_message(message: str) -> None:
    """Add a custom opportunity message (can use {discipline} and {discipline_field} placeholders)."""
    _custom_messages.append(message)


def get_random_message_with_custom() -> str:
    """Get a random message including any custom messages."""
    all_templates = OPPORTUNITY_MESSAGE_TEMPLATES + _custom_messages
    template = random.choice(all_templates)
    return _format_message(template)


def clear_custom_messages() -> None:
    """Clear all custom messages (useful for testing)."""
    global _custom_messages
    _custom_messages = []


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests() -> bool:
    """Create unit tests for opportunity_messages module."""
    from test_framework import TestSuite

    suite = TestSuite("Opportunity Messages Tests", "opportunity_messages.py")
    suite.start_suite()

    def test_get_random_opportunity_message():
        msg = get_random_opportunity_message()
        assert msg is not None
        assert isinstance(msg, str)
        assert len(msg) > 0
        # Message should be a formatted version of a template
        all_msgs = get_all_messages()
        assert msg in all_msgs

    def test_get_opportunity_message_by_index_valid():
        msg = get_opportunity_message_by_index(0)
        expected = _format_message(OPPORTUNITY_MESSAGE_TEMPLATES[0])
        assert msg == expected
        msg = get_opportunity_message_by_index(49)
        expected = _format_message(OPPORTUNITY_MESSAGE_TEMPLATES[49])
        assert msg == expected

    def test_get_opportunity_message_by_index_wraps():
        # Index wraps around via modulo
        msg = get_opportunity_message_by_index(50)
        expected = _format_message(OPPORTUNITY_MESSAGE_TEMPLATES[0])
        assert msg == expected
        msg = get_opportunity_message_by_index(51)
        expected = _format_message(OPPORTUNITY_MESSAGE_TEMPLATES[1])
        assert msg == expected

    def test_get_all_messages():
        msgs = get_all_messages()
        assert isinstance(msgs, list)
        assert len(msgs) == 50
        # Verify it's a new list each time
        msgs.append("test")
        assert len(get_all_messages()) == 50

    def test_get_message_count():
        count = get_message_count()
        assert count == 50

    def test_add_custom_message():
        clear_custom_messages()  # Reset first
        add_custom_message("Custom test message")
        # Custom message should appear in random with custom
        all_with_custom = OPPORTUNITY_MESSAGE_TEMPLATES + _custom_messages
        assert "Custom test message" in all_with_custom
        clear_custom_messages()  # Cleanup

    def test_get_random_message_with_custom():
        clear_custom_messages()  # Reset first
        add_custom_message("Unique custom message XYZ123")
        # Verify custom message is in the combined list
        all_templates = OPPORTUNITY_MESSAGE_TEMPLATES + _custom_messages
        assert "Unique custom message XYZ123" in all_templates, (
            f"Custom message not in list. _custom_messages={_custom_messages}"
        )
        # Verify get_random_message_with_custom returns from combined list
        msg = get_random_message_with_custom()
        all_formatted = [_format_message(t) for t in all_templates]
        assert msg in all_formatted, "Returned message should be from combined list"
        clear_custom_messages()  # Cleanup

    def test_all_messages_have_ps_prefix():
        for msg in get_all_messages():
            assert msg.startswith("P.S."), (
                f"Message should start with 'P.S.': {msg[:30]}"
            )

    suite.run_test(
        test_name="Get random opportunity message",
        test_func=test_get_random_opportunity_message,
        test_summary="Tests Get random opportunity message functionality",
        method_description="Calls get random opportunity message and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Get message by index - valid",
        test_func=test_get_opportunity_message_by_index_valid,
        test_summary="Tests Get message by index with valid scenario",
        method_description="Calls get opportunity message by index and verifies the result",
        expected_outcome="Function returns the expected successful result",
    )
    suite.run_test(
        test_name="Get message by index - wraps",
        test_func=test_get_opportunity_message_by_index_wraps,
        test_summary="Tests Get message by index with wraps scenario",
        method_description="Calls get opportunity message by index and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Get all messages",
        test_func=test_get_all_messages,
        test_summary="Tests Get all messages functionality",
        method_description="Calls get all messages and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Get message count",
        test_func=test_get_message_count,
        test_summary="Tests Get message count functionality",
        method_description="Calls get message count and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Add custom message",
        test_func=test_add_custom_message,
        test_summary="Tests Add custom message functionality",
        method_description="Calls clear custom messages and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Get random with custom",
        test_func=test_get_random_message_with_custom,
        test_summary="Tests Get random with custom functionality",
        method_description="Calls clear custom messages and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="All messages have P.S. prefix",
        test_func=test_all_messages_have_ps_prefix,
        test_summary="Tests All messages have P.S. prefix functionality",
        method_description="Calls get all messages and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )

    return suite.finish_suite()


run_comprehensive_tests = __import__("test_framework").create_standard_test_runner(
    _create_module_tests
)
