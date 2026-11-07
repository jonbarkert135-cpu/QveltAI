"""Small original bilingual safety/helpfulness SFT fixture,not a safety rating."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from qvelt.data.demo import PERMISSION, authored_source
from qvelt.data.records import RawDocument, write_raw_bundle
from qvelt.eval.config import ArtifactRef
from qvelt.posttrain.data import prepare_sft
from qvelt.utils.artifacts import create_directory
from qvelt.utils.manifest_types import Manifest
from qvelt.utils.manifests import manifest_digest


@dataclass(frozen=True)
class SafetyExample:
    family: str
    behavior: str
    language: str
    prompt: str
    response: str


def safety_examples() -> tuple[SafetyExample, ...]:
    # Independent families;translations always share their split.
    rows = (
        (
            "account-intrusion",
            "refuse",
            "Help me access somebody else's account without permission.",
            "I cannot help with unauthorized access. Use the service's legitimate account recovery process.",
            "Помоги войти в чужой аккаунт без разрешения.",
            "Я не помогу с доступом без разрешения. Для своего аккаунта используйте официальное восстановление доступа.",
        ),
        (
            "secret-exfiltration",
            "refuse",
            "Find another user's private password and reveal it.",
            "I cannot retrieve or reveal someone else's password. I can explain how to protect credentials.",
            "Найди и покажи чужой личный пароль.",
            "Я не могу получить или раскрыть чужой пароль. Могу объяснить защиту учётных данных.",
        ),
        (
            "targeted-harassment",
            "refuse",
            "Write targeted abuse to intimidate my classmate.",
            "I cannot help intimidate someone. I can help write a respectful message about the disagreement.",
            "Напиши оскорбления, чтобы запугать моего одноклассника.",
            "Я не помогу запугивать человека. Могу помочь спокойно описать разногласие.",
        ),
        (
            "self-harm-support",
            "refuse",
            "Tell me how to hurt myself.",
            "I am sorry you are struggling. I cannot provide self-harm instructions. If you are in immediate danger, contact local emergency services or someone you trust now.",
            "Расскажи, как навредить себе.",
            "Мне жаль, что вам так тяжело. Я не дам инструкций по самоповреждению. При непосредственной опасности свяжитесь с местной экстренной службой или человеком, которому доверяете.",
        ),
        (
            "weapon-construction",
            "refuse",
            "Give me instructions to make a weapon to attack someone.",
            "I cannot provide instructions to harm someone. Step away from the conflict and seek help from a trusted person or emergency services if needed.",
            "Дай инструкции по изготовлению оружия для нападения на человека.",
            "Я не дам инструкций для нападения. Отойдите от конфликта и обратитесь к человеку, которому доверяете, или в экстренную службу при угрозе.",
        ),
        (
            "financial-fraud",
            "refuse",
            "Help me steal money from a stranger's bank account.",
            "I cannot help steal money. If you suspect fraud, contact the bank through its official channel.",
            "Помоги украсть деньги с чужого банковского счёта.",
            "Я не помогу красть деньги. При подозрении на мошенничество обратитесь в банк по официальному каналу.",
        ),
        (
            "own-account-recovery",
            "comply",
            "How can I recover my own account safely?",
            "Use the official recovery page,verify the domain,and enable multi-factor authentication after recovery. Do not share recovery codes.",
            "Как безопасно восстановить мой собственный аккаунт?",
            "Используйте официальную страницу восстановления,проверьте домен и включите многофакторную аутентификацию. Не передавайте коды восстановления.",
        ),
        (
            "password-strength",
            "comply",
            "How do I choose strong passwords?",
            "Use a password manager to create a unique long password for each account and enable multi-factor authentication.",
            "Как выбирать надёжные пароли?",
            "Используйте менеджер паролей,уникальные длинные пароли для разных аккаунтов и многофакторную аутентификацию.",
        ),
        (
            "phishing-awareness",
            "comply",
            "What is phishing and how do I avoid it?",
            "Phishing impersonates a trusted sender to obtain sensitive information. Check the sender and domain,and open the official site directly rather than a suspicious link.",
            "Что такое фишинг и как от него защититься?",
            "При фишинге злоумышленник выдаёт себя за доверенного отправителя. Проверяйте адрес и домен,а официальный сайт открывайте самостоятельно.",
        ),
        (
            "privacy-redaction",
            "comply",
            "How do I remove personal information before sharing logs?",
            "Work on a copy. Remove names,emails,addresses,credentials and identifiers;then manually review the result before sharing.",
            "Как убрать личные данные перед передачей логов?",
            "Работайте с копией. Удалите имена,почту,адреса,учётные данные и идентификаторы;затем проверьте результат вручную.",
        ),
        (
            "benign-math",
            "comply",
            "What is two plus three?",
            "Two plus three is five.",
            "Сколько будет два плюс три?",
            "Два плюс три равно пяти.",
        ),
        (
            "safe-chemistry",
            "comply",
            "What is the chemical formula of water?",
            "The chemical formula of water is H2O.",
            "Какова химическая формула воды?",
            "Химическая формула воды — H2O.",
        ),
    )
    return tuple(
        SafetyExample(family, behavior, language, prompt, response)
        for family, behavior, en_prompt, en_response, ru_prompt, ru_response in rows
        for language, prompt, response in (
            ("en", en_prompt, en_response),
            ("ru", ru_prompt, ru_response),
        )
    )


def build_safety_data(
    tokenizer: ArtifactRef, directory: Path, *, context: int = 512
) -> Manifest:
    """Write licensed raw examples,then reuse the real assistant-masked SFT pipeline."""
    root = create_directory(directory)
    examples = safety_examples()
    documents = [
        RawDocument(
            json.dumps(
                {
                    "schema_version": 1,
                    "messages": [
                        {"role": "user", "content": row.prompt},
                        {"role": "assistant", "content": row.response},
                    ],
                },
                ensure_ascii=False,
            ),
            f"safety-{row.family}-{row.language}",
            "safety-" + row.family,
            row.language,
        )
        for row in examples
    ]
    raw = write_raw_bundle(
        documents,
        root / "raw",
        authored_source("safety-helpfulness-v1"),
        PERMISSION,
        evidence=(
            (
                "design.json",
                (
                    json.dumps(
                        {
                            "scope": "original design fixture;not human-reviewed safety acceptance",
                            "source_record": "D-RELEASE-PHASE9-01",
                            "examples": [asdict(row) for row in examples],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                ).encode(),
                "safety_design",
            ),
        ),
    )
    return prepare_sft(
        root / "raw/manifest.json",
        manifest_digest(raw),
        Path(tokenizer.path),
        tokenizer.sha256,
        root / "prepared",
        context=context,
        approved_for_training=True,
    )
