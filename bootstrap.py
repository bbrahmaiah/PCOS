from jarvis.runtime.config import get_settings


def main() -> None:
    settings = get_settings()

    print(
        f"{settings.runtime.system_name} "
        f"{settings.runtime.version} ONLINE"
    )


if __name__ == "__main__":
    main()