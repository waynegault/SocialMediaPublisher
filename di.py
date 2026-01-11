"""Dependency Injection and Service Container for Social Media Publisher.

This module provides a service container and dependency injection framework
for better testability, modularity, and configuration management.

Features:
- Service container with lazy instantiation
- Interface-based design patterns
- Factory pattern for component creation
- Environment-based configuration (dev/test/prod)
- Easy mocking for tests

Example:
    container = ServiceContainer()
    container.register("database", Database)
    container.register("publisher", LinkedInPublisher, database=Ref("database"))
    
    publisher = container.get("publisher")
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    from test_framework import TestSuite


# =============================================================================
# Environment Configuration
# =============================================================================


class Environment(Enum):
    """Application environment types."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class EnvironmentConfig:
    """Configuration for each environment."""

    name: Environment
    debug: bool = False
    log_level: str = "INFO"
    mock_external: bool = False
    database_path: str = ""
    cache_enabled: bool = True

    @classmethod
    def development(cls) -> EnvironmentConfig:
        """Create development configuration."""
        return cls(
            name=Environment.DEVELOPMENT,
            debug=True,
            log_level="DEBUG",
            mock_external=False,
            database_path=":memory:",
            cache_enabled=True,
        )

    @classmethod
    def testing(cls) -> EnvironmentConfig:
        """Create testing configuration."""
        return cls(
            name=Environment.TESTING,
            debug=True,
            log_level="WARNING",
            mock_external=True,
            database_path=":memory:",
            cache_enabled=False,
        )

    @classmethod
    def production(cls) -> EnvironmentConfig:
        """Create production configuration."""
        return cls(
            name=Environment.PRODUCTION,
            debug=False,
            log_level="INFO",
            mock_external=False,
            database_path="content_engine.db",
            cache_enabled=True,
        )


# =============================================================================
# Service Protocols (Interfaces)
# =============================================================================


@runtime_checkable
class Disposable(Protocol):
    """Protocol for services that need cleanup."""

    def close(self) -> None:
        """Clean up resources."""
        ...


@runtime_checkable
class Initializable(Protocol):
    """Protocol for services that need initialization."""

    def initialize(self) -> None:
        """Initialize the service."""
        ...


T = TypeVar("T")


class ServiceProtocol(ABC, Generic[T]):
    """Base protocol for all services."""

    @abstractmethod
    def get_instance(self) -> T:
        """Get the service instance."""
        pass


# =============================================================================
# Service Registration
# =============================================================================


@dataclass
class Ref:
    """Reference to another service by name."""

    name: str


@dataclass
class ServiceRegistration:
    """Registration info for a service."""

    name: str
    factory: Callable[..., Any]
    singleton: bool = True
    lazy: bool = True
    dependencies: dict[str, Ref | Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    environment: Environment | None = None  # None = all environments


class ServiceScope(Enum):
    """Scope of service lifetime."""

    SINGLETON = "singleton"  # One instance for container lifetime
    TRANSIENT = "transient"  # New instance each time
    SCOPED = "scoped"  # One instance per scope


# =============================================================================
# Service Container
# =============================================================================


class ServiceContainer:
    """Dependency injection container."""

    def __init__(self, environment: Environment = Environment.DEVELOPMENT) -> None:
        """Initialize the container.

        Args:
            environment: Current environment
        """
        self.environment = environment
        self._registrations: dict[str, ServiceRegistration] = {}
        self._instances: dict[str, Any] = {}
        self._building: set[str] = set()  # For circular dependency detection
        self._parent: ServiceContainer | None = None

    def register(
        self,
        name: str,
        factory: Callable[..., Any] | type,
        *,
        singleton: bool = True,
        lazy: bool = True,
        tags: list[str] | None = None,
        environment: Environment | None = None,
        **dependencies: Ref | Any,
    ) -> ServiceContainer:
        """Register a service.

        Args:
            name: Service name
            factory: Factory function or class
            singleton: If True, only one instance is created
            lazy: If True, instance is created on first access
            tags: Optional tags for grouping
            environment: Optional environment restriction
            **dependencies: Named dependencies (use Ref for other services)

        Returns:
            Self for chaining
        """
        self._registrations[name] = ServiceRegistration(
            name=name,
            factory=factory,
            singleton=singleton,
            lazy=lazy,
            dependencies=dependencies,
            tags=tags or [],
            environment=environment,
        )
        return self

    def register_instance(self, name: str, instance: Any) -> ServiceContainer:
        """Register an existing instance.

        Args:
            name: Service name
            instance: Service instance

        Returns:
            Self for chaining
        """
        self._instances[name] = instance
        self._registrations[name] = ServiceRegistration(
            name=name,
            factory=lambda: instance,
            singleton=True,
            lazy=False,
        )
        return self

    def register_factory(
        self,
        name: str,
        factory: Callable[[ServiceContainer], Any],
        singleton: bool = True,
    ) -> ServiceContainer:
        """Register a factory that receives the container.

        Args:
            name: Service name
            factory: Factory function that takes the container
            singleton: If True, only one instance is created

        Returns:
            Self for chaining
        """
        self._registrations[name] = ServiceRegistration(
            name=name,
            factory=lambda: factory(self),
            singleton=singleton,
            lazy=True,
        )
        return self

    def get(self, name: str) -> Any:
        """Get a service by name.

        Args:
            name: Service name

        Returns:
            Service instance

        Raises:
            KeyError: If service not found
            RuntimeError: If circular dependency detected
        """
        # Check if already instantiated
        if name in self._instances:
            return self._instances[name]

        # Check parent container
        if name not in self._registrations and self._parent:
            return self._parent.get(name)

        if name not in self._registrations:
            raise KeyError(f"Service not registered: {name}")

        registration = self._registrations[name]

        # Check environment restriction
        if registration.environment and registration.environment != self.environment:
            raise KeyError(
                f"Service {name} not available in {self.environment.value} environment"
            )

        # Detect circular dependencies
        if name in self._building:
            raise RuntimeError(f"Circular dependency detected: {name}")

        self._building.add(name)

        try:
            # Resolve dependencies
            resolved_deps = {}
            for dep_name, dep_value in registration.dependencies.items():
                if isinstance(dep_value, Ref):
                    resolved_deps[dep_name] = self.get(dep_value.name)
                else:
                    resolved_deps[dep_name] = dep_value

            # Create instance
            instance = registration.factory(**resolved_deps)

            # Initialize if needed
            if isinstance(instance, Initializable):
                instance.initialize()

            # Store if singleton
            if registration.singleton:
                self._instances[name] = instance

            return instance

        finally:
            self._building.discard(name)

    def has(self, name: str) -> bool:
        """Check if a service is registered.

        Args:
            name: Service name

        Returns:
            True if registered
        """
        if name in self._registrations:
            return True
        if self._parent:
            return self._parent.has(name)
        return False

    def get_by_tag(self, tag: str) -> list[Any]:
        """Get all services with a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of service instances
        """
        services = []
        for name, reg in self._registrations.items():
            if tag in reg.tags:
                services.append(self.get(name))
        return services

    def create_child_scope(self) -> ServiceContainer:
        """Create a child container scope.

        Returns:
            Child container
        """
        child = ServiceContainer(self.environment)
        child._parent = self
        return child

    def override(self, name: str, instance: Any) -> ServiceContainer:
        """Override a service with a different instance (useful for testing).

        Args:
            name: Service name
            instance: Override instance

        Returns:
            Self for chaining
        """
        self._instances[name] = instance
        return self

    def reset(self, name: str | None = None) -> None:
        """Reset service instances.

        Args:
            name: Optional specific service to reset (None = all)
        """
        if name:
            self._instances.pop(name, None)
        else:
            # Dispose all disposable services
            for instance in self._instances.values():
                if isinstance(instance, Disposable):
                    instance.close()
            self._instances.clear()

    def get_registration_info(self) -> dict[str, dict[str, Any]]:
        """Get info about all registrations.

        Returns:
            Dictionary of registration details
        """
        return {
            name: {
                "singleton": reg.singleton,
                "lazy": reg.lazy,
                "tags": reg.tags,
                "environment": reg.environment.value if reg.environment else None,
                "instantiated": name in self._instances,
            }
            for name, reg in self._registrations.items()
        }


# =============================================================================
# Service Provider Pattern
# =============================================================================


class ServiceProvider(ABC):
    """Base class for modules that register services."""

    @abstractmethod
    def register(self, container: ServiceContainer) -> None:
        """Register services in the container.

        Args:
            container: The service container
        """
        pass

    def boot(self, container: ServiceContainer) -> None:
        """Boot the provider after all services are registered.

        Args:
            container: The service container
        """
        pass


class CoreServicesProvider(ServiceProvider):
    """Provider for core application services."""

    def register(self, container: ServiceContainer) -> None:
        """Register core services."""
        # Register environment config
        if container.environment == Environment.PRODUCTION:
            container.register_instance("config", EnvironmentConfig.production())
        elif container.environment == Environment.TESTING:
            container.register_instance("config", EnvironmentConfig.testing())
        else:
            container.register_instance("config", EnvironmentConfig.development())


# =============================================================================
# Application Builder
# =============================================================================


class ApplicationBuilder:
    """Builder for configuring and creating application container."""

    def __init__(self) -> None:
        """Initialize the builder."""
        self._environment = Environment.DEVELOPMENT
        self._providers: list[ServiceProvider] = []
        self._registrations: list[tuple[str, Callable[..., Any], dict[str, Any]]] = []

    def with_environment(self, environment: Environment) -> ApplicationBuilder:
        """Set the environment.

        Args:
            environment: Target environment

        Returns:
            Self for chaining
        """
        self._environment = environment
        return self

    def add_provider(self, provider: ServiceProvider) -> ApplicationBuilder:
        """Add a service provider.

        Args:
            provider: Service provider

        Returns:
            Self for chaining
        """
        self._providers.append(provider)
        return self

    def add_service(
        self,
        name: str,
        factory: Callable[..., Any],
        **options: Any,
    ) -> ApplicationBuilder:
        """Add a service registration.

        Args:
            name: Service name
            factory: Factory function
            **options: Registration options

        Returns:
            Self for chaining
        """
        self._registrations.append((name, factory, options))
        return self

    def build(self) -> ServiceContainer:
        """Build the container.

        Returns:
            Configured ServiceContainer
        """
        container = ServiceContainer(self._environment)

        # Register from providers
        for provider in self._providers:
            provider.register(container)

        # Register individual services
        for name, factory, options in self._registrations:
            container.register(name, factory, **options)

        # Boot providers
        for provider in self._providers:
            provider.boot(container)

        return container


# =============================================================================
# Convenience Functions
# =============================================================================

# Global container instance
_container: ServiceContainer | None = None


def get_container() -> ServiceContainer:
    """Get the global container instance.

    Returns:
        Global ServiceContainer
    """
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container


def set_container(container: ServiceContainer) -> None:
    """Set the global container instance.

    Args:
        container: Container to use globally
    """
    global _container
    _container = container


def inject(name: str) -> Any:
    """Get a service from the global container.

    Args:
        name: Service name

    Returns:
        Service instance
    """
    return get_container().get(name)


# =============================================================================
# Unit Tests
# =============================================================================


def _create_module_tests() -> "TestSuite":
    """Create unit tests for this module."""
    sys.path.insert(0, str(Path(__file__).parent))
    from test_framework import TestSuite

    suite = TestSuite("Dependency Injection")

    def test_environment_config():
        config = EnvironmentConfig.development()
        assert config.name == Environment.DEVELOPMENT
        assert config.debug is True

    def test_environment_production():
        config = EnvironmentConfig.production()
        assert config.debug is False
        assert config.log_level == "INFO"

    def test_container_init():
        container = ServiceContainer()
        assert container.environment == Environment.DEVELOPMENT

    def test_container_register():
        container = ServiceContainer()
        container.register("test", lambda: "value")
        assert container.has("test")

    def test_container_get():
        container = ServiceContainer()
        container.register("test", lambda: "value")
        assert container.get("test") == "value"

    def test_container_singleton():
        container = ServiceContainer()
        call_count = [0]

        def factory() -> dict[str, int]:
            call_count[0] += 1
            return {"count": call_count[0]}

        container.register("test", factory, singleton=True)
        result1 = container.get("test")
        result2 = container.get("test")
        assert result1 is result2
        assert call_count[0] == 1

    def test_container_transient():
        container = ServiceContainer()
        call_count = [0]

        def factory() -> dict[str, int]:
            call_count[0] += 1
            return {"count": call_count[0]}

        container.register("test", factory, singleton=False)
        result1 = container.get("test")
        result2 = container.get("test")
        assert result1 is not result2
        assert call_count[0] == 2

    def test_container_dependencies():
        container = ServiceContainer()
        container.register("db", lambda: {"type": "database"})
        container.register("service", lambda db: {"db": db}, db=Ref("db"))

        service = container.get("service")
        assert service["db"]["type"] == "database"

    def test_container_circular_detection():
        container = ServiceContainer()
        container.register("a", lambda b: b, b=Ref("b"))
        container.register("b", lambda a: a, a=Ref("a"))

        try:
            container.get("a")
            assert False, "Should raise RuntimeError"
        except RuntimeError as e:
            assert "Circular" in str(e)

    def test_container_register_instance():
        container = ServiceContainer()
        instance = {"pre": "configured"}
        container.register_instance("test", instance)
        assert container.get("test") is instance

    def test_container_override():
        container = ServiceContainer()
        container.register("test", lambda: "original")
        container.override("test", "override")
        assert container.get("test") == "override"

    def test_container_tags():
        container = ServiceContainer()
        container.register("svc1", lambda: 1, tags=["api"])
        container.register("svc2", lambda: 2, tags=["api"])
        container.register("svc3", lambda: 3, tags=["other"])

        api_services = container.get_by_tag("api")
        assert len(api_services) == 2

    def test_application_builder():
        builder = ApplicationBuilder()
        container = builder.with_environment(Environment.TESTING).build()
        assert container.environment == Environment.TESTING

    def test_service_provider():
        class TestProvider(ServiceProvider):
            def register(self, container: ServiceContainer) -> None:
                container.register("from_provider", lambda: "provided")

        builder = ApplicationBuilder()
        builder.add_provider(TestProvider())
        container = builder.build()
        assert container.get("from_provider") == "provided"

    suite.add_test("Environment config", test_environment_config)
    suite.add_test("Environment production", test_environment_production)
    suite.add_test("Container init", test_container_init)
    suite.add_test("Container register", test_container_register)
    suite.add_test("Container get", test_container_get)
    suite.add_test("Container singleton", test_container_singleton)
    suite.add_test("Container transient", test_container_transient)
    suite.add_test("Container dependencies", test_container_dependencies)
    suite.add_test("Circular detection", test_container_circular_detection)
    suite.add_test("Register instance", test_container_register_instance)
    suite.add_test("Override service", test_container_override)
    suite.add_test("Service tags", test_container_tags)
    suite.add_test("Application builder", test_application_builder)
    suite.add_test("Service provider", test_service_provider)

    return suite


if __name__ == "__main__":
    # Demo usage
    print("Dependency Injection Demo")
    print("=" * 50)

    # Build application container
    container = (
        ApplicationBuilder()
        .with_environment(Environment.DEVELOPMENT)
        .add_provider(CoreServicesProvider())
        .add_service("greeting", lambda: "Hello, World!")
        .build()
    )

    print(f"\nEnvironment: {container.environment.value}")
    print(f"Greeting: {container.get('greeting')}")

    config = container.get("config")
    print(f"Debug mode: {config.debug}")

    # Show registrations
    print("\nRegistered services:")
    for name, info in container.get_registration_info().items():
        print(f"  - {name}: singleton={info['singleton']}")
