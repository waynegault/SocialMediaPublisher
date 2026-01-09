"""
Professional job opportunity postscript messages for LinkedIn posts.

These messages convey openness to new opportunities in a non-begging,
professional, and engaging manner suitable for a chemical engineering professional.
"""

import random

# 50 professional postscript messages about job opportunities
OPPORTUNITY_MESSAGES = [
    # Direct but professional
    "P.S. I'm currently exploring new opportunities in chemical engineering. Let's connect!",
    "P.S. Open to discussing exciting chemical engineering roles. Feel free to reach out!",
    "P.S. I'm actively seeking my next challenge in the chemical engineering space.",
    "P.S. Looking for my next opportunity to make an impact in chemical engineering.",
    "P.S. Currently exploring new chapters in my chemical engineering career.",
    # Expertise-focused
    "P.S. I bring process optimization expertise to every role. Open to new opportunities!",
    "P.S. Seeking roles where I can apply my chemical engineering skills to real-world challenges.",
    "P.S. Looking to bring my technical expertise to an innovative team. Let's talk!",
    "P.S. Ready to contribute my engineering experience to a forward-thinking organization.",
    "P.S. Eager to apply my process engineering background to new challenges.",
    # Value proposition
    "P.S. If your team needs someone who can bridge technical and business needs, I'd love to chat.",
    "P.S. I help organizations optimize their chemical processes. Currently available for new roles.",
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
    "P.S. Passionate about sustainable chemical processes. Open to green chemistry opportunities!",
    "P.S. Seeking roles in process safety, optimization, or plant operations.",
    "P.S. Looking for opportunities in petrochemical, pharmaceutical, or specialty chemicals.",
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
    "P.S. Always up for a conversation about chemical engineering. And yes, I'm job hunting!",
    "P.S. Let's discuss the future of chemical engineering. I'm also exploring new opportunities.",
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
    "P.S. Excited about where chemical engineering is heading. Looking to be part of it!",
    "P.S. The industry is evolving rapidly. I want to help shape its future.",
    "P.S. Seeking roles at the cutting edge of chemical engineering.",
    "P.S. Ready to contribute to the next generation of process innovation.",
    "P.S. Looking for opportunities to work on tomorrow's engineering challenges.",
]


def get_random_opportunity_message() -> str:
    """Get a randomly selected opportunity message."""
    return random.choice(OPPORTUNITY_MESSAGES)


def get_opportunity_message_by_index(index: int) -> str:
    """Get a specific opportunity message by index (0-49)."""
    return OPPORTUNITY_MESSAGES[index % len(OPPORTUNITY_MESSAGES)]


def get_all_messages() -> list[str]:
    """Get all opportunity messages."""
    return OPPORTUNITY_MESSAGES.copy()


def get_message_count() -> int:
    """Get the total number of available messages."""
    return len(OPPORTUNITY_MESSAGES)


# Optional: Allow customization via config
_custom_messages: list[str] = []


def add_custom_message(message: str) -> None:
    """Add a custom opportunity message."""
    _custom_messages.append(message)


def get_random_message_with_custom() -> str:
    """Get a random message including any custom messages."""
    all_msgs = OPPORTUNITY_MESSAGES + _custom_messages
    return random.choice(all_msgs)


def clear_custom_messages() -> None:
    """Clear all custom messages (useful for testing)."""
    global _custom_messages
    _custom_messages = []


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for opportunity_messages module."""
    from test_framework import TestSuite

    suite = TestSuite("Opportunity Messages Tests")

    def test_get_random_opportunity_message():
        msg = get_random_opportunity_message()
        assert msg is not None
        assert isinstance(msg, str)
        assert len(msg) > 0
        assert msg in OPPORTUNITY_MESSAGES

    def test_get_opportunity_message_by_index_valid():
        msg = get_opportunity_message_by_index(0)
        assert msg == OPPORTUNITY_MESSAGES[0]
        msg = get_opportunity_message_by_index(49)
        assert msg == OPPORTUNITY_MESSAGES[49]

    def test_get_opportunity_message_by_index_wraps():
        # Index wraps around via modulo
        msg = get_opportunity_message_by_index(50)
        assert msg == OPPORTUNITY_MESSAGES[0]
        msg = get_opportunity_message_by_index(51)
        assert msg == OPPORTUNITY_MESSAGES[1]

    def test_get_all_messages():
        msgs = get_all_messages()
        assert isinstance(msgs, list)
        assert len(msgs) == 50
        # Verify it's a copy
        msgs.append("test")
        assert len(get_all_messages()) == 50

    def test_get_message_count():
        count = get_message_count()
        assert count == 50

    def test_add_custom_message():
        clear_custom_messages()  # Reset first
        add_custom_message("Custom test message")
        # Custom message should appear in random with custom
        all_with_custom = OPPORTUNITY_MESSAGES + _custom_messages
        assert "Custom test message" in all_with_custom
        clear_custom_messages()  # Cleanup

    def test_get_random_message_with_custom():
        clear_custom_messages()  # Reset first
        add_custom_message("Unique custom message XYZ123")
        # Get many random messages to verify custom can appear
        found_custom = False
        for _ in range(200):  # Should find it within 200 tries
            msg = get_random_message_with_custom()
            if "Unique custom message XYZ123" in msg:
                found_custom = True
                break
        assert found_custom, "Custom message should appear in random selection"
        clear_custom_messages()  # Cleanup

    def test_all_messages_have_ps_prefix():
        for msg in OPPORTUNITY_MESSAGES:
            assert msg.startswith("P.S."), (
                f"Message should start with 'P.S.': {msg[:30]}"
            )

    suite.add_test(
        "Get random opportunity message", test_get_random_opportunity_message
    )
    suite.add_test(
        "Get message by index - valid", test_get_opportunity_message_by_index_valid
    )
    suite.add_test(
        "Get message by index - wraps", test_get_opportunity_message_by_index_wraps
    )
    suite.add_test("Get all messages", test_get_all_messages)
    suite.add_test("Get message count", test_get_message_count)
    suite.add_test("Add custom message", test_add_custom_message)
    suite.add_test("Get random with custom", test_get_random_message_with_custom)
    suite.add_test("All messages have P.S. prefix", test_all_messages_have_ps_prefix)

    return suite
