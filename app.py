import matplotlib
matplotlib.use("Agg")

import os
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import joblib
import pickle
import spacy
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

try:
    from sparsify import Sae
    SPARSIFY_IMPORT_ERROR = None
except Exception as e:
    Sae = None
    SPARSIFY_IMPORT_ERROR = e
import xgboost as xgb

class TimingCallback(xgb.callback.TrainingCallback):
    def __init__(self, total_rounds=None, print_every=25):
        self.total_rounds = total_rounds
        self.print_every = print_every

    def before_training(self, model):
        return model

    def after_iteration(self, model, epoch, evals_log):
        return False

    def after_training(self, model):
        return model

# Initialize models (DependencyAI & DivEye)
class DependencyAIDetector:
    def __init__(self, vectorizer_path, model_path):
        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)
        self.nlp = spacy.load("ru_core_news_lg")

    def extract_dependency_sequence(self, text):
        doc = self.nlp(text)
        dep_seq = " ".join([token.dep_ for token in doc])
        return dep_seq

    def predict_proba(self, text):
        dep_seq = self.extract_dependency_sequence(text)
        transformed_text = self.vectorizer.transform([dep_seq])
        return _predict_dependency_ai_proba(self.model, transformed_text)



DEPENDENCY_LABEL_EXPLANATIONS = {
    "dep": {
        "short": "dep",
        "label": "неуточнённая зависимость",
        "description": "парсер не смог выбрать более точный тип связи; такой сигнал требует осторожной интерпретации"
    },
    "nmod": {
        "short": "nmod",
        "label": "номинальный модификатор",
        "description": "одно существительное уточняет другое существительное"
    },
    "root": {
        "short": "root",
        "label": "корневой узел предложения",
        "description": "главный предикат или центр синтаксической структуры"
    },
    "ccomp": {
        "short": "ccomp",
        "label": "комплементное придаточное",
        "description": "придаточная часть передаёт содержание мысли, речи или оценки"
    },
    "det": {
        "short": "det",
        "label": "детерминатив",
        "description": "уточняющее слово вроде указателя или определителя"
    },
    "advmod": {
        "short": "advmod",
        "label": "обстоятельственный модификатор",
        "description": "слово уточняет действие или признак по способу, степени или времени"
    },
    "obl": {
        "short": "obl",
        "label": "косвенный именной компонент",
        "description": "именная группа при сказуемом, задающая контекст места, времени, причины и т.д."
    },
    "punct": {
        "short": "punct",
        "label": "пунктуационный элемент",
        "description": "знак препинания как часть синтаксической структуры"
    },
    "cc": {
        "short": "cc",
        "label": "сочинительный союз",
        "description": "союз вроде «и», «а», «но», соединяющий однородные элементы"
    },
    "conj": {
        "short": "conj",
        "label": "сочинительная связь",
        "description": "связь между однородными элементами или частями конструкции"
    },
    "nsubj": {
        "short": "nsubj",
        "label": "подлежащее",
        "description": "носитель действия или состояния"
    },
    "obj": {
        "short": "obj",
        "label": "прямое дополнение",
        "description": "объект действия"
    },
    "case": {
        "short": "case",
        "label": "падежный/предложный маркер",
        "description": "служебное слово, маркирующее падежную связь"
    },
    "amod": {
        "short": "amod",
        "label": "определение-прилагательное",
        "description": "прилагательное уточняет существительное"
    },
    "appos": {
        "short": "appos",
        "label": "приложение",
        "description": "поясняющая именная группа рядом с другим существительным"
    },
    "flat:foreign": {
        "short": "flat:foreign",
        "label": "иностранный плоский фрагмент",
        "description": "неразложимое иностранное словосочетание"
    },
}


def _sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _dependency_pattern_to_display(pattern):
    tags = [t for t in pattern.split() if t]
    if not tags:
        return {
            "exact_pattern": pattern,
            "readable_pattern": pattern,
            "explanation": "",
        }

    label_parts = []
    explanation_parts = []
    for tag in tags:
        info = DEPENDENCY_LABEL_EXPLANATIONS.get(tag)
        if info is None:
            label_parts.append(tag)
            explanation_parts.append(f"{tag}: служебный синтаксический маркер")
        else:
            label_parts.append(f"{tag} — {info['label']}")
            explanation_parts.append(f"{tag}: {info['description']}")

    readable_pattern = " → ".join(label_parts) if len(label_parts) > 1 else label_parts[0]
    explanation = "; ".join(explanation_parts)
    return {
        "exact_pattern": pattern,
        "readable_pattern": readable_pattern,
        "explanation": explanation,
    }


def _matrix_to_1d_array(X):
    if hasattr(X, "toarray"):
        return np.asarray(X.toarray(), dtype=np.float32)[0]
    return np.asarray(X, dtype=np.float32).reshape(-1)


AI_LABEL_HINTS = ("ai", "machine", "generated", "synthetic", "llm", "gpt", "bot")
HUMAN_LABEL_HINTS = ("human", "человек", "author", "real")


def _infer_ai_class_index(clf):
    classes = getattr(clf, "classes_", None)
    if classes is None:
        return 1, {"mode": "default", "classes": None}

    labels = [str(c).strip().lower() for c in classes]
    for idx, label in enumerate(labels):
        if any(hint in label for hint in AI_LABEL_HINTS):
            return idx, {"mode": "label_hint", "classes": list(classes)}

    for idx, label in enumerate(labels):
        if any(hint in label for hint in HUMAN_LABEL_HINTS) and len(labels) == 2:
            return 1 - idx, {"mode": "inverse_human_hint", "classes": list(classes)}

    return 1, {"mode": "default", "classes": list(classes)}


def _predict_dependency_ai_proba(clf, X):
    proba = np.asarray(clf.predict_proba(X), dtype=np.float64)
    ai_idx, _ = _infer_ai_class_index(clf)

    if proba.ndim == 1:
        flat = proba.reshape(-1)
        if flat.size == 1:
            return float(flat[0])
        if flat.size >= 2:
            ai_idx = min(ai_idx, flat.size - 1)
            return float(flat[ai_idx])

    if proba.ndim == 2:
        ai_idx = min(ai_idx, proba.shape[1] - 1)
        return float(proba[0, ai_idx])

    flat = proba.reshape(-1)
    ai_idx = min(ai_idx, max(0, flat.size - 1))
    return float(flat[ai_idx])


def _estimate_dependency_feature_impacts(clf, X, feature_names, tfidf_values, prob_ai, max_features=40):
    active_idx = np.flatnonzero(tfidf_values > 0)
    if active_idx.size == 0:
        return []

    if active_idx.size > max_features:
        order = np.argsort(tfidf_values[active_idx])[::-1]
        active_idx = active_idx[order[:max_features]]

    X_csr = X.tocsr() if hasattr(X, "tocsr") else X
    rows = []

    for feat_idx in active_idx:
        X_mod = X_csr.copy()
        if hasattr(X_mod, "tocsr"):
            X_mod = X_mod.tocsr()
            row_start, row_end = X_mod.indptr[0], X_mod.indptr[1]
            local_pos = np.where(X_mod.indices[row_start:row_end] == feat_idx)[0]
            if local_pos.size == 0:
                continue
            data_pos = row_start + int(local_pos[0])
            original_value = float(X_mod.data[data_pos])
            X_mod.data[data_pos] = 0.0
            X_mod.eliminate_zeros()
        else:
            X_mod = np.asarray(X_mod).copy()
            if feat_idx >= X_mod.shape[1]:
                continue
            original_value = float(X_mod[0, feat_idx])
            X_mod[0, feat_idx] = 0.0

        try:
            prob_without = _predict_dependency_ai_proba(clf, X_mod)
        except Exception:
            continue

        impact_pp = (float(prob_ai) - float(prob_without)) * 100.0
        display = _dependency_pattern_to_display(feature_names[feat_idx])
        rows.append({
            "pattern": feature_names[feat_idx],
            "exact_pattern": display["exact_pattern"],
            "readable_pattern": display["readable_pattern"],
            "explanation": display["explanation"],
            "tfidf_value": float(tfidf_values[feat_idx]),
            "original_feature_value": original_value,
            "prob_without_ai_pct": float(prob_without) * 100.0,
            "impact_pp": float(impact_pp),
            "abs_impact_pp": abs(float(impact_pp)),
            "direction": "ai" if float(impact_pp) > 0 else "human",
        })

    total_abs = float(sum(row["abs_impact_pp"] for row in rows))
    for row in rows:
        row["share_abs_pct"] = (row["abs_impact_pp"] / total_abs * 100.0) if total_abs > 0 else 0.0

    rows.sort(key=lambda row: row["abs_impact_pp"], reverse=True)
    return rows


def _get_dependency_local_contributions(clf, X):
    contrib_values = None
    bias = 0.0
    mode = "approx"

    # LightGBM sklearn API
    try:
        contrib = clf.predict(X, pred_contrib=True)
        contrib = np.asarray(contrib, dtype=np.float32)
        if contrib.ndim == 2 and contrib.shape[0] == 1 and contrib.shape[1] >= 2:
            contrib_values = contrib[0, :-1]
            bias = float(contrib[0, -1])
            mode = "pred_contrib"
    except TypeError:
        pass
    except Exception:
        pass

    # booster_ fallback
    if contrib_values is None and hasattr(clf, "booster_"):
        try:
            contrib = clf.booster_.predict(X, pred_contrib=True)
            contrib = np.asarray(contrib, dtype=np.float32)
            if contrib.ndim == 2 and contrib.shape[0] == 1 and contrib.shape[1] >= 2:
                contrib_values = contrib[0, :-1]
                bias = float(contrib[0, -1])
                mode = "pred_contrib"
        except Exception:
            pass

    if contrib_values is None:
        tfidf_values = _matrix_to_1d_array(X)
        importances = getattr(clf, "feature_importances_", None)
        if importances is None:
            importances = np.ones_like(tfidf_values, dtype=np.float32)
        else:
            importances = np.asarray(importances, dtype=np.float32)
            if importances.shape[0] != tfidf_values.shape[0]:
                importances = np.resize(importances, tfidf_values.shape[0])
        contrib_values = tfidf_values * importances

    raw_score = None
    if mode == "pred_contrib":
        raw_score = float(np.sum(contrib_values) + bias)
    else:
        try:
            raw = clf.predict(X, raw_score=True)
            raw_score = float(np.asarray(raw).reshape(-1)[0])
        except Exception:
            raw_score = None

    return np.asarray(contrib_values, dtype=np.float32), bias, mode, raw_score


def _build_dependency_feature_rows(feature_names, tfidf_values, contrib_values, prob_ai, raw_score=None):
    rows = []
    total_abs = float(np.sum(np.abs(contrib_values)))
    for feature_name, tfidf_value, contrib in zip(feature_names, tfidf_values, contrib_values):
        contrib = float(contrib)
        if float(tfidf_value) <= 0 and abs(contrib) < 1e-10:
            continue

        display = _dependency_pattern_to_display(feature_name)
        delta_prob_pp = None
        prob_without = None
        if raw_score is not None:
            prob_without = float(_sigmoid(raw_score - contrib))
            delta_prob_pp = (float(prob_ai) - prob_without) * 100.0

        odds_multiplier = float(np.exp(min(abs(contrib), 12.0)))

        rows.append({
            "pattern": feature_name,
            "exact_pattern": display["exact_pattern"],
            "readable_pattern": display["readable_pattern"],
            "explanation": display["explanation"],
            "tfidf_value": float(tfidf_value),
            "contribution": contrib,
            "abs_contribution": abs(contrib),
            "share_abs_pct": (abs(contrib) / total_abs * 100.0) if total_abs > 0 else 0.0,
            "delta_prob_pp": delta_prob_pp,
            "prob_without_ai_pct": (prob_without * 100.0) if prob_without is not None else None,
            "odds_multiplier": odds_multiplier,
            "direction": "ai" if contrib > 0 else ("human" if contrib < 0 else "neutral"),
        })

    rows.sort(key=lambda row: abs(row["contribution"]), reverse=True)
    return rows




def _build_dependency_active_patterns(feature_names, tfidf_values, limit=6):
    active_idx = np.flatnonzero(tfidf_values > 0)
    if active_idx.size == 0:
        return []

    order = active_idx[np.argsort(tfidf_values[active_idx])[::-1]]
    selected = order[:limit]
    total_selected = float(np.sum(tfidf_values[selected])) if selected.size else 0.0

    rows = []
    for feat_idx in selected:
        feature_name = feature_names[feat_idx]
        display = _dependency_pattern_to_display(feature_name)
        token_count = len([t for t in feature_name.split() if t])
        rows.append({
            "pattern": feature_name,
            "exact_pattern": display["exact_pattern"],
            "readable_pattern": display["readable_pattern"],
            "explanation": display["explanation"],
            "tfidf_value": float(tfidf_values[feat_idx]),
            "prominence_pct": (float(tfidf_values[feat_idx]) / total_selected * 100.0) if total_selected > 0 else 0.0,
            "ngram_order": token_count,
            "ngram_label": "одиночная связь" if token_count == 1 else f"цепочка из {token_count} связей",
        })
    return rows

def _dependency_reliability(doc):
    tokens = [t for t in doc if not t.is_space]
    token_count = len(tokens)
    sent_count = len(list(doc.sents))
    dep_ratio = 0.0
    if token_count:
        dep_ratio = sum(1 for t in tokens if t.dep_ == "dep") / token_count

    score = 0
    reasons = []

    if token_count >= 120:
        score += 2
        reasons.append("текст достаточно длинный для устойчивого dependency-анализа")
    elif token_count >= 60:
        score += 1
        reasons.append("длина текста приемлема, но не максимальна для синтаксического анализа")
    else:
        reasons.append("текст короткий, поэтому синтаксические сигналы менее устойчивы")

    if sent_count >= 4:
        score += 1
        reasons.append("в тексте несколько предложений, поэтому решение опирается не на один фрагмент")
    else:
        reasons.append("предложений мало, часть вывода может зависеть от одного-двух фрагментов")

    if dep_ratio <= 0.08:
        score += 1
        reasons.append("доля неуточнённых связей dep невысокая, парсер увереннее в разборе")
    elif dep_ratio <= 0.15:
        reasons.append("доля неуточнённых связей dep умеренная")
    else:
        reasons.append("доля неуточнённых связей dep повышенная, интерпретацию нужно читать осторожно")

    if score >= 4:
        label = "высокая"
    elif score >= 2:
        label = "средняя"
    else:
        label = "ограниченная"

    return {
        "label": label,
        "token_count": token_count,
        "sentence_count": sent_count,
        "dep_ratio_pct": dep_ratio * 100.0,
        "reasons": reasons,
    }


def _select_dependency_sentence_examples(doc, vectorizer, clf, overall_is_ai, limit=4):
    sentence_rows = []
    sentence_texts = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        word_count = len([t for t in sent if not t.is_punct and not t.is_space])
        sentence_texts.append((sent_text, word_count, sent))

    filtered = [(txt, wc, sent) for txt, wc, sent in sentence_texts if wc >= 5]
    if not filtered:
        filtered = sentence_texts

    feature_names = vectorizer.get_feature_names_out()

    for sent_text, word_count, sent in filtered:
        dep_seq = " ".join(token.dep_ for token in sent if not token.is_space)
        if not dep_seq.strip():
            continue
        X_sent = vectorizer.transform([dep_seq])
        try:
            sent_prob = _predict_dependency_ai_proba(clf, X_sent)
        except Exception:
            continue
        tfidf_values = _matrix_to_1d_array(X_sent)
        sent_rows = _estimate_dependency_feature_impacts(clf, X_sent, feature_names, tfidf_values, sent_prob, max_features=12)

        if overall_is_ai:
            best_row = next((row for row in sent_rows if row["impact_pp"] > 0), sent_rows[0] if sent_rows else None)
            support_margin = sent_prob - 0.5
        else:
            best_row = next((row for row in sent_rows if row["impact_pp"] < 0), sent_rows[0] if sent_rows else None)
            support_margin = 0.5 - sent_prob

        if best_row is None:
            continue

        sentence_rows.append({
            "sentence": sent_text if len(sent_text) <= 260 else sent_text[:257] + "...",
            "word_count": word_count,
            "prob_ai_pct": sent_prob * 100.0,
            "support_margin_pct": max(0.0, support_margin * 100.0),
            "top_pattern": best_row["exact_pattern"],
            "top_pattern_readable": best_row["readable_pattern"],
            "top_pattern_impact_pp": best_row["impact_pp"],
        })

    sentence_rows.sort(key=lambda row: row["support_margin_pct"], reverse=True)
    return sentence_rows[:limit]


class RussianAIDetector:
    def __init__(self, model_path="./local_model", xgb_path="diveye_llmtrace_ru_booster.pkl"):
        self.device = "cpu"
        self.max_length = 1024

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()

        if xgb_path.endswith(".joblib"):
            artifact = joblib.load(xgb_path)
        else:
            with open(xgb_path, "rb") as f:
                artifact = pickle.load(f)

        if hasattr(artifact, "predict_proba"):
            self.clf = artifact
            self.calibrator = None
            self.threshold = 0.5
            self.feature_columns = None
            self.use_booster = False

        elif isinstance(artifact, dict):
            self.clf = artifact["model"]
            self.calibrator = artifact.get("calibrator")
            self.threshold = float(artifact.get("threshold", 0.5))
            self.feature_columns = artifact.get("feature_columns")
            self.use_booster = bool(artifact.get("use_booster", False))
        else:
            raise ValueError("Неизвестный формат файла детектора")

        self.feature_dim = len(self.feature_columns) if self.feature_columns else 9
        print("Модель и токенизатор загружены из локальной папки:", model_path)
        print("DivEye threshold =", self.threshold)
        print("DivEye feature_dim =", self.feature_dim)

    @torch.no_grad()
    def _compute_surprisal(self, text, max_length=None):
        max_length = max_length or self.max_length
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.device)

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        surprisal = -token_log_probs.detach().cpu().numpy()[0]
        return surprisal.astype(np.float32)

    def _safe_quantile(self, arr, q):
        if arr.size == 0:
            return 0.0
        return float(np.quantile(arr, q))

    def _safe_mean(self, arr):
        return float(np.mean(arr)) if arr.size else 0.0

    def _safe_std(self, arr):
        return float(np.std(arr)) if arr.size else 0.0

    def _safe_var(self, arr):
        return float(np.var(arr)) if arr.size else 0.0

    def _safe_max(self, arr):
        return float(np.max(arr)) if arr.size else 0.0

    def _text_stabilizer_features(self, text, surprisal):
        token_count = int(len(surprisal))

        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        sent_lengths = [len(s.split()) for s in sentences if s.strip()]

        mean_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
        std_sent_len = float(np.std(sent_lengths)) if sent_lengths else 0.0

        punct_ratio = 0.0
        if len(text) > 0:
            punct_ratio = len(re.findall(r"[,:;.!?\-()\"]", text)) / max(len(text), 1)

        words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if words:
            ttr = len(set(words)) / len(words)
        else:
            ttr = 0.0

        return {
            "token_count": float(token_count),
            "mean_sent_len": float(mean_sent_len),
            "std_sent_len": float(std_sent_len),
            "punct_ratio": float(punct_ratio),
            "ttr": float(ttr),
        }

    def _extract_features_dict(self, text, surprisal):
        S = np.asarray(surprisal, dtype=np.float32)
        if S.size == 0:
            base = {
                "s_mean": 0.0, "s_std": 0.0, "s_q90": 0.0,
                "d1_mean_abs": 0.0, "d1_std": 0.0, "d1_q90_abs": 0.0,
                "d2_mean_abs": 0.0, "d2_std": 0.0, "d2_q90_abs": 0.0,
            }
            base.update(self._text_stabilizer_features(text, S))
            return base

        d1 = np.diff(S)
        d2 = np.diff(d1)

        feats = {
            "s_mean": self._safe_mean(S),
            "s_std": self._safe_std(S),
            "s_q90": self._safe_quantile(S, 0.90),

            "d1_mean_abs": self._safe_mean(np.abs(d1)),
            "d1_std": self._safe_std(d1),
            "d1_q90_abs": self._safe_quantile(np.abs(d1), 0.90),

            "d2_mean_abs": self._safe_mean(np.abs(d2)),
            "d2_std": self._safe_std(d2),
            "d2_q90_abs": self._safe_quantile(np.abs(d2), 0.90),
        }

        feats.update(self._text_stabilizer_features(text, S))
        return feats

    def _extract_features(self, surprisal, text=None):
        text = text or ""
        feats_dict = self._extract_features_dict(text, surprisal)

        if self.feature_columns:
            return np.array(
                [float(feats_dict.get(col, 0.0)) for col in self.feature_columns],
                dtype=np.float32
            )

        S = np.asarray(surprisal, dtype=np.float32)
        if S.size == 0:
            return np.zeros(9, dtype=np.float32)

        dS = np.diff(S)
        d2S = np.diff(dS)

        def safe_stats(arr):
            if arr.size == 0:
                return [0.0, 0.0, 0.0]
            return [float(np.mean(arr)), float(np.var(arr)), float(np.max(arr))]

        feats = safe_stats(S) + safe_stats(dS) + safe_stats(d2S)
        return np.array(feats, dtype=np.float32)

    def _predict_raw_proba(self, X):
        proba = float(self.clf.predict_proba(X)[0, 1])

        if self.calibrator is not None:
            try:
                proba = float(self.calibrator.predict(np.array([proba]))[0])
            except Exception:
                pass

        return proba

    def predict_proba(self, text):
        try:
            if not text or len(text.strip()) == 0:
                return 0.5, "Не определено", 0.5

            surprisal_seq = self._compute_surprisal(text)
            features = self._extract_features(surprisal_seq, text=text)
            proba_ai = self._predict_raw_proba(features.reshape(1, -1))

            label = "ИИ-ГЕНЕРИРОВАННЫЙ" if proba_ai >= self.threshold else "Человеческий"

            if proba_ai >= self.threshold:
                confidence = proba_ai
            else:
                confidence = 1.0 - proba_ai

            return proba_ai, label, confidence

        except Exception as e:
            print(f"Ошибка при детекции ИИ: {e}")
            return 0.5, "Не определено", 0.5



class SAEDeepSeekXGBDetector:
    def __init__(
        self,
        deepseek_root="deepseek",
        config_path=None,
        xgb_path=None,
        model_path=None,
        sae_path=None,
    ):
        self.device = "cpu"
        self.available = False
        self.load_error = None

        self.deepseek_root = deepseek_root
        self.config_path = config_path
        self.xgb_path = xgb_path
        self.model_path = model_path
        self.sae_path = sae_path

        self.model_name = None
        self.sae_repo = None
        self.max_length = 768
        self.batch_size = 8
        self.layer = None
        self.hookpoint_name = None

        try:
            if Sae is None:
                raise ImportError(
                    "Не удалось импортировать пакет sparsify. "
                    "Установите eai-sparsify: pip install eai-sparsify"
                ) from SPARSIFY_IMPORT_ERROR

            resolved = self._resolve_paths(
                deepseek_root=deepseek_root,
                config_path=config_path,
                xgb_path=xgb_path,
                model_path=model_path,
                sae_path=sae_path,
            )

            self.config_path = resolved["config_path"]
            self.xgb_path = resolved["xgb_path"]
            self.model_source = resolved["model_path"]
            self.sae_source = resolved["sae_path"]

            self._load_config(self.config_path)

            local_only = os.path.isdir(self.model_source)
            common_kwargs = {"local_files_only": local_only}

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_source,
                use_fast=True,
                **common_kwargs,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

            model_kwargs = dict(common_kwargs)
            model_kwargs["torch_dtype"] = torch.float32

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_source,
                    low_cpu_mem_usage=True,
                    **model_kwargs,
                ).to(self.device)
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_source,
                    **model_kwargs,
                ).to(self.device)

            self.model.eval()

            self.sae = Sae.load_from_disk(self.sae_source, device=self.device)
            self.sae = self.sae.to(self.device)
            self.sae.eval()

            self.clf = joblib.load(self.xgb_path)
            self.available = True
            print(
                "SAE/DeepSeek/XGB detector loaded:",
                f"layer={self.layer}, hookpoint={self.hookpoint_name}, "
                f"model={self.model_source}, sae={self.sae_source}"
            )
        except Exception as e:
            self.load_error = str(e)
            print(f"SAE/DeepSeek/XGB detector unavailable: {e}")

    def _resolve_paths(self, deepseek_root, config_path, xgb_path, model_path, sae_path):
        root = Path(deepseek_root)

        resolved_config = Path(config_path) if config_path else root / "artifacts" / "run_config.json"

        if xgb_path:
            resolved_xgb = Path(xgb_path)
        else:
            candidates = sorted((root / "artifacts").glob("xgb_layer_*.joblib"))
            if not candidates:
                raise FileNotFoundError(
                    "Не найден XGBoost-файл для DeepSeek. "
                    "Ожидается deepseek/artifacts/xgb_layer_<layer>.joblib"
                )
            resolved_xgb = candidates[0]

        if model_path:
            resolved_model = Path(model_path)
        else:
            model_candidates = [
                root / "DeepSeek-R1-Distill-Qwen-1.5B",
                root / "model",
            ]
            resolved_model = None
            for candidate in model_candidates:
                if candidate.exists():
                    resolved_model = candidate
                    break
            if resolved_model is None:
                raise FileNotFoundError(
                    "Не найдена локальная папка модели DeepSeek. "
                    "Ожидается deepseek/DeepSeek-R1-Distill-Qwen-1.5B"
                )

        if sae_path:
            resolved_sae = Path(sae_path)
        else:
            sae_candidates = sorted(root.glob("sae_layer_*"))
            if not sae_candidates:
                raise FileNotFoundError(
                    "Не найдена локальная папка SAE для DeepSeek. "
                    "Ожидается deepseek/sae_layer_<layer>"
                )
            resolved_sae = sae_candidates[0]

        return {
            "config_path": str(resolved_config),
            "xgb_path": str(resolved_xgb),
            "model_path": str(resolved_model),
            "sae_path": str(resolved_sae),
        }

    def _load_config(self, config_path):
        cfg = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

        self.model_name = cfg.get("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        self.sae_repo = cfg.get("SAE_REPO")
        self.max_length = int(cfg.get("MAX_LENGTH", 768))
        self.batch_size = int(cfg.get("BATCH_SIZE", 8))

        if self.layer is None:
            layers_to_run = cfg.get("LAYERS_TO_RUN") or []
            if layers_to_run:
                self.layer = int(layers_to_run[0])

        if self.layer is None:
            xgb_match = re.search(r"xgb_layer_(\d+)\.joblib$", str(self.xgb_path))
            if xgb_match:
                self.layer = int(xgb_match.group(1))

        if self.layer is None:
            sae_match = re.search(r"sae_layer_(\d+)$", str(self.sae_path).rstrip("/\\"))
            if sae_match:
                self.layer = int(sae_match.group(1))

        if self.layer is None:
            raise ValueError("Не удалось определить слой DeepSeek по конфигу и именам файлов.")

        self.hookpoint_name = f"layers.{self.layer}.mlp"

    def _tokenize(self, texts):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.model.device) for k, v in enc.items()}

    @torch.no_grad()
    def _encode_dense_latents(self, hidden_states):
        B, T, D = hidden_states.shape
        flat = hidden_states.reshape(B * T, D)

        top_acts, top_indices, pre_acts = self.sae.encode(flat)

        dense_latents = torch.zeros(
            flat.shape[0],
            self.sae.num_latents,
            device=flat.device,
            dtype=top_acts.dtype,
        )
        dense_latents.scatter_(1, top_indices, top_acts)

        sae_latents = dense_latents.reshape(B, T, self.sae.num_latents)
        return sae_latents

    @torch.no_grad()
    def _extract_features(self, texts, return_latents=False):
        batch = self._tokenize(texts)
        outputs = self.model(**batch, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # Важно: здесь сохранено то же приближение, что и в обучающем ноутбуке.
        # Для hookpoint layers.<n>.mlp используется hidden_states[layer].
        h = hidden_states[self.layer]

        sae_latents = self._encode_dense_latents(h)

        attn = batch["attention_mask"].unsqueeze(-1).to(sae_latents.dtype)
        sae_latents = sae_latents * attn

        pooled = sae_latents.sum(dim=1)
        X = pooled.detach().float().cpu().numpy()

        if return_latents:
            return X, sae_latents, batch
        return X

    def predict_proba(self, text):
        if not self.available:
            return None
        try:
            if not text or not text.strip():
                return 0.5, "Не определено", 0.5

            X = self._extract_features([text])
            proba_ai = float(self.clf.predict_proba(X)[0, 1])
            label = "ИИ-ГЕНЕРИРОВАННЫЙ" if proba_ai > 0.5 else "Человеческий"
            confidence = proba_ai if proba_ai > 0.5 else (1 - proba_ai)
            return proba_ai, label, confidence
        except Exception as e:
            print(f"Ошибка при SAE/DeepSeek/XGB детекции: {e}")
            return None

from dataclasses import dataclass


@dataclass
class EnsembleVerdict:
    final_ai_prob: float
    final_human_prob: float
    final_label: str
    confidence_pct: float
    diveye_weight: float
    dependency_weight: float
    rationale: list[str]


def ensemble_ai_verdict(
    dependency_ai_prob: float,
    diveye_ai_prob: float,
    text_length_tokens: int,
    has_repetition: bool = False,
    has_anomalous_tail: bool = False,
    surprisal_is_smooth: bool = False,
    syntactic_is_too_regular: bool = False,
    need_second_opinion: bool = True,
) -> EnsembleVerdict:
    dependency_ai_prob = max(0.0, min(1.0, dependency_ai_prob))
    diveye_ai_prob = max(0.0, min(1.0, diveye_ai_prob))

    diveye_weight = 0.70
    dependency_weight = 0.30
    rationale: list[str] = []

    dependency_says_human = dependency_ai_prob < 0.5
    dependency_human_conf = 1.0 - dependency_ai_prob

    if has_repetition:
        diveye_weight += 0.05
        dependency_weight -= 0.05
        rationale.append("Обнаружены повторы или однотипные фрагменты: увеличен вес DivEye.")

    if has_anomalous_tail:
        diveye_weight += 0.05
        dependency_weight -= 0.05
        rationale.append("На графике surprisal обнаружен аномальный хвост: увеличен вес DivEye.")

    if surprisal_is_smooth:
        diveye_weight += 0.03
        dependency_weight -= 0.03
        rationale.append("График surprisal выглядит сглаженным и повторяемым: увеличен вес DivEye.")

    if text_length_tokens >= 250:
        diveye_weight += 0.03
        dependency_weight -= 0.03
        rationale.append("Текст достаточно длинный для устойчивого ритмического анализа DivEye.")

    if dependency_says_human and dependency_human_conf < 0.75:
        diveye_weight += 0.02
        dependency_weight -= 0.02
        rationale.append("DependencyAI склоняется к человеку без высокой уверенности: вес DivEye увеличен.")

    diveye_weight = min(diveye_weight, 0.85)
    dependency_weight = max(1.0 - diveye_weight, 0.15)

    if text_length_tokens < 120:
        dependency_weight += 0.10
        diveye_weight -= 0.10
        rationale.append("Текст короткий: DivEye может быть менее стабилен, поэтому увеличен вес DependencyAI.")

    if syntactic_is_too_regular:
        dependency_weight += 0.08
        diveye_weight -= 0.08
        rationale.append("Синтаксис выглядит слишком регулярным: увеличен вес DependencyAI.")

    if need_second_opinion:
        rationale.append("DependencyAI сохранён как второе мнение другого типа, не завязанное на surprisal.")

    total_w = diveye_weight + dependency_weight
    diveye_weight /= total_w
    dependency_weight /= total_w

    final_ai_prob = dependency_weight * dependency_ai_prob + diveye_weight * diveye_ai_prob
    final_human_prob = 1.0 - final_ai_prob

    if final_ai_prob >= 0.5:
        final_label = "ИИ"
        confidence_pct = final_ai_prob * 100.0
    else:
        final_label = "Человек"
        confidence_pct = final_human_prob * 100.0

    disagreement = abs(dependency_ai_prob - diveye_ai_prob)
    if disagreement >= 0.40:
        rationale.append("Методы существенно расходятся в оценке; итоговый вердикт следует интерпретировать осторожно.")

    return EnsembleVerdict(
        final_ai_prob=final_ai_prob,
        final_human_prob=final_human_prob,
        final_label=final_label,
        confidence_pct=confidence_pct,
        diveye_weight=diveye_weight,
        dependency_weight=dependency_weight,
        rationale=rationale,
    )


def _detect_repetition(text: str) -> bool:
    paragraphs = [p.strip().lower() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) >= 2:
        unique_ratio = len(set(paragraphs)) / len(paragraphs)
        if unique_ratio < 0.8:
            return True

    sentences = [s.strip().lower() for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(sentences) >= 6:
        unique_ratio = len(set(sentences)) / len(sentences)
        if unique_ratio < 0.85:
            return True

    return False


def _analyze_surprisal_profile(detector, text: str) -> dict:
    surprisal_seq = detector._compute_surprisal(text)
    s = np.asarray(surprisal_seq, dtype=np.float32)

    if s.size == 0:
        return {
            "has_anomalous_tail": False,
            "surprisal_is_smooth": False,
            "text_length_tokens": 0,
        }

    thirds = np.array_split(s, 3)
    first_mean = float(np.mean(thirds[0])) if len(thirds[0]) else 0.0
    last_mean = float(np.mean(thirds[-1])) if len(thirds[-1]) else 0.0

    tail = thirds[-1] if len(thirds[-1]) else s
    near_zero_tail_ratio = float(np.mean(tail < 0.5)) if tail.size else 0.0
    has_anomalous_tail = (first_mean > 0 and last_mean < first_mean * 0.45) or (near_zero_tail_ratio > 0.55)

    std_s = float(np.std(s))
    d1 = np.diff(s)
    smooth_change = float(np.std(d1)) if d1.size else 0.0
    surprisal_is_smooth = std_s < 1.2 or smooth_change < 0.9

    return {
        "has_anomalous_tail": has_anomalous_tail,
        "surprisal_is_smooth": surprisal_is_smooth,
        "text_length_tokens": int(len(s)),
    }


def _syntactic_is_too_regular(detector, text: str) -> bool:
    dep_seq = detector.extract_dependency_sequence(text)
    tags = dep_seq.split()
    if not tags:
        return False

    from collections import Counter
    counts = Counter(tags)
    most_common_ratio = counts.most_common(1)[0][1] / len(tags)
    unique_ratio = len(counts) / len(tags)

    return most_common_ratio > 0.22 or unique_ratio < 0.12

# Combined model class
class CombinedAIDetector:
    def __init__(self, dependency_model, diveye_model, sae_xgb_model=None):
        self.dependency_model = dependency_model
        self.diveye_model = diveye_model
        self.sae_xgb_model = sae_xgb_model

    def predict(self, text):
        prob_dependency = self.dependency_model.predict_proba(text)

        diveye_ai_prob, label_diveye, conf_diveye = self.diveye_model.predict_proba(text)

        repetition_flag = _detect_repetition(text)
        surprisal_meta = _analyze_surprisal_profile(self.diveye_model, text)
        syntactic_regular = _syntactic_is_too_regular(self.dependency_model, text)
        
        sae_result = self.sae_xgb_model.predict_proba(text) if self.sae_xgb_model else None
        sae_available = sae_result is not None
        sae_ai_prob = float(sae_result[0]) if sae_available else None

        ensemble = ensemble_ai_verdict(
        dependency_ai_prob=prob_dependency,
        diveye_ai_prob=diveye_ai_prob,
            text_length_tokens=surprisal_meta["text_length_tokens"],
            has_repetition=repetition_flag,
            has_anomalous_tail=surprisal_meta["has_anomalous_tail"],
            surprisal_is_smooth=surprisal_meta["surprisal_is_smooth"],
            syntactic_is_too_regular=syntactic_regular,
            need_second_opinion=True,
        )

        base_ai_prob = float(ensemble.final_ai_prob)
        rationale = list(ensemble.rationale)

        if sae_available:
            final_ai_prob = 0.6 * sae_ai_prob + 0.4 * base_ai_prob
            rationale.insert(0, "В итог также добавлен SAE/DeepSeek/XGB детектор, обученный на признаках SAE латентов DeepSeek.")
            rationale.append("Итоговая вероятность = 60% SAE/DeepSeek/XGB + 40% ансамбль DependencyAI/DivEye.")
            sae_weight = 0.60
            legacy_weight = 0.40
        else:
            final_ai_prob = base_ai_prob
            sae_weight = 0.0
            legacy_weight = 1.0
            rationale.append("SAE/DeepSeek/XGB детектор недоступен, поэтому итог построен только на DependencyAI и DivEye.")

        final_human_prob = 1.0 - final_ai_prob
        if final_ai_prob >= 0.5:
            final_label = "ИИ"
            confidence_pct = final_ai_prob * 100.0
        else:
            final_label = "Человек"
            confidence_pct = final_human_prob * 100.0

        return {
            "probability_dependencyAI": prob_dependency,
            "probability_divEye": diveye_ai_prob,
            "probability_sae_xgb": sae_ai_prob,

            "legacy_average_probability": base_ai_prob,
            "average_probability": final_ai_prob,

            "final_prediction": final_label,
            "final_confidence": confidence_pct,

            "diveye_weight": ensemble.diveye_weight,
            "dependency_weight": ensemble.dependency_weight,

            "sae_xgb_available": sae_available,
            "sae_xgb_error": getattr(self.sae_xgb_model, 'load_error', None) if self.sae_xgb_model else None,
            "sae_xgb_weight": sae_weight,
            "legacy_ensemble_weight": legacy_weight,

            "ensemble_rationale": rationale,
}

# Функция для расширенного анализа DependencyAI
def extended_analysis(text_or_file_path, detector=None, vectorizer_path='dependency_vectorizer.pkl', model_path='dependency_model.pkl'):
    is_path = isinstance(text_or_file_path, str) and os.path.exists(text_or_file_path)
    if is_path:
        with open(text_or_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = str(text_or_file_path or '')

    if not content.strip():
        return {"error": "Недостаточно текста для DependencyAI-анализа."}

    try:
        if detector is None:
            detector = DependencyAIDetector(vectorizer_path, model_path)
        vectorizer = detector.vectorizer
        clf = detector.model
    except FileNotFoundError:
        return {"error": "Ошибка: модели DependencyAI не найдены."}
    except Exception as e:
        return {"error": f"Ошибка загрузки DependencyAI: {str(e)}"}

    dep_seq = detector.extract_dependency_sequence(content)
    tfidf_matrix = vectorizer.transform([dep_seq])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_values = _matrix_to_1d_array(tfidf_matrix)

    prob = _predict_dependency_ai_proba(clf, tfidf_matrix)
    is_ai = prob >= 0.5
    confidence = prob if is_ai else (1.0 - prob)
    verdict_text = "Текст сгенерирован с помощью ИИ" if is_ai else "Текст написан человеком"

    prob_ai_pct = prob * 100.0
    prob_human_pct = (1.0 - prob) * 100.0
    margin_pct = abs(prob - 0.5) * 100.0
    margin_side = "в пользу ИИ" if is_ai else "в пользу человеческого текста"

    if confidence >= 0.95:
        signal_strength = "высокая"
    elif confidence >= 0.80:
        signal_strength = "средняя"
    else:
        signal_strength = "умеренная"

    active_patterns = _build_dependency_active_patterns(feature_names, tfidf_values, limit=6)

    img_path = None

    doc = detector.nlp(content)
    reliability = _dependency_reliability(doc)

    description_text = (
        "DependencyAI анализирует не сами слова, а последовательность синтаксических связей в тексте. "
        "Текст сначала проходит dependency-разбор, затем связи переводятся в TF–IDF признаки, "
        "после чего классификатор оценивает вероятность ИИ. "
        "Подробнее про метод можно почитать здесь: "
        '<a href="https://arxiv.org/pdf/2602.15514" target="_blank" rel="noopener">ссылка на статью</a>.'
    )

    method_note = (
        "DependencyAI принимает решение по совокупности синтаксических паттернов текста, а не по одному отдельному признаку."
    )

    if active_patterns:
        patterns_note = (
            "Ниже показаны наиболее заметные dependency-паттерны, которые встретились в текущем тексте."
        )
    else:
        patterns_note = "Не удалось выделить заметные dependency-паттерны для показа."

    glossary_tags = []
    seen = set()
    for row in active_patterns:
        for tag in [tag for tag in row["pattern"].split() if tag]:
            if tag in seen:
                continue
            seen.add(tag)
            info = DEPENDENCY_LABEL_EXPLANATIONS.get(tag)
            if info is None:
                glossary_tags.append({
                    "tag": tag,
                    "label": "служебный маркер",
                    "description": "редкий или нерасшифрованный тип синтаксической связи",
                })
            else:
                glossary_tags.append({
                    "tag": tag,
                    "label": info["label"],
                    "description": info["description"],
                })

    return {
        "verdict_text": verdict_text,
        "confidence_pct": confidence * 100.0,
        "prob_ai_pct": prob_ai_pct,
        "prob_human_pct": prob_human_pct,
        "threshold_pct": 50.0,
        "margin_pct": margin_pct,
        "margin_side": margin_side,
        "signal_strength": signal_strength,
        "description_text": description_text,
        "method_note": method_note,
        "patterns_note": patterns_note,
        "img_path": img_path,
        "analysis_mode": "compact_probability_and_patterns",
        "reliability": reliability,
        "active_patterns": active_patterns,
        "glossary_tags": glossary_tags,
    }

def extended_analysis_diveye(text, detector, diveye_ai_prob):
    if not text or not text.strip():
        return {
            "verdict_text": "Недостаточно текста для анализа",
            "confidence_pct": 0.0,
            "description_text": "",
            "img_path": None,
            "group_cards": [],
            "top_signals": [],
            "summary_text": ""
        }

    surprisal_seq = detector._compute_surprisal(text)
    features = detector._extract_features(surprisal_seq, text=text)

    score = float(diveye_ai_prob)
    is_ai = score >= getattr(detector, "threshold", 0.5)
    confidence_pct = (score if is_ai else (1.0 - score)) * 100.0
    verdict_text = "Текст сгенерирован с помощью ИИ" if is_ai else "Текст написан человеком"

    description_text = """
    Метод DivEye анализирует не сами слова, а то, как по ходу текста меняется неожиданность токенов для языковой модели.
    В этой версии также учитываются дополнительные стабилизирующие признаки текста и применяется калибровка вероятности.
    Подробнее про метод можно почитать здесь:
    <a href="https://arxiv.org/pdf/2509.18880" target="_blank" rel="noopener">ссылка на статью</a>.
    """

    if hasattr(detector.clf, "feature_importances_"):
        importances = np.asarray(detector.clf.feature_importances_, dtype=np.float32)
    else:
        importances = np.ones(len(features), dtype=np.float32)

    local_scores = np.abs(features) * importances
    total_local = float(local_scores.sum()) if local_scores.size else 0.0

    if detector.feature_columns:
        feature_names = detector.feature_columns
    else:
        feature_names = [
            "s_mean", "s_var", "s_max",
            "d1_mean", "d1_var", "d1_max",
            "d2_mean", "d2_var", "d2_max",
        ]

    readable_map = {
        "s_mean": "Средняя неожиданность текста",
        "s_std": "Разброс неожиданности текста",
        "s_q90": "Высокие пики неожиданности",
        "d1_mean_abs": "Средняя сила локальных изменений",
        "d1_std": "Разброс локальных изменений",
        "d1_q90_abs": "Сильные локальные скачки",
        "d2_mean_abs": "Средняя глубинная неровность",
        "d2_std": "Разброс глубинной неровности",
        "d2_q90_abs": "Сильные вторичные колебания",
        "token_count": "Длина текста в токенах",
        "mean_sent_len": "Средняя длина предложений",
        "std_sent_len": "Разброс длины предложений",
        "punct_ratio": "Доля пунктуации",
        "ttr": "Лексическое разнообразие",
        "base_score": "Оценка базового детектора",
    }

    explanation_map = {
        "s_mean": "Показывает, насколько текст в целом предсказуем для языковой модели.",
        "s_std": "Показывает, насколько текст ровный или неоднородный по уровню неожиданности.",
        "s_q90": "Отражает выраженные неожиданные участки текста.",
        "d1_mean_abs": "Характеризует среднюю силу локальных изменений ритма.",
        "d1_std": "Показывает, насколько неравномерно меняется ритм текста.",
        "d1_q90_abs": "Выделяет сильные локальные переломы ритма.",
        "d2_mean_abs": "Характеризует среднюю глубинную изменчивость ритма.",
        "d2_std": "Показывает, насколько неоднородна глубинная динамика текста.",
        "d2_q90_abs": "Фиксирует самые сильные вторичные колебания ритма.",
        "token_count": "Длинные и короткие тексты ведут себя по-разному; длина помогает стабилизировать решение.",
        "mean_sent_len": "Средняя длина предложений помогает учитывать общий стиль текста.",
        "std_sent_len": "Разброс длины предложений показывает естественность структуры.",
        "punct_ratio": "Доля пунктуации помогает учитывать форму изложения.",
        "ttr": "Лексическое разнообразие помогает учитывать богатство словаря.",
        "base_score": "Сигнал вашего базового детектора, усиленный DivEye.",
    }

    feature_items = []
    for name, raw in zip(feature_names, local_scores):
        share_pct = (float(raw) / total_local * 100.0) if total_local > 0 else 0.0
        feature_items.append({
            "name": readable_map.get(name, name),
            "share_pct": share_pct,
            "explanation": explanation_map.get(name, "Дополнительный признак, влияющий на итоговое решение.")
        })

    feature_items = sorted(feature_items, key=lambda x: x["share_pct"], reverse=True)
    top_signals = feature_items[:5]

    # Группировка
    group_scores = {
        "Ритм surprisal": 0.0,
        "Стабилизаторы текста": 0.0,
        "Базовый детектор": 0.0,
    }

    for name, raw in zip(feature_names, local_scores):
        if name.startswith("s_") or name.startswith("d1_") or name.startswith("d2_"):
            group_scores["Ритм surprisal"] += float(raw)
        elif name == "base_score":
            group_scores["Базовый детектор"] += float(raw)
        else:
            group_scores["Стабилизаторы текста"] += float(raw)

    total_groups = sum(group_scores.values()) or 1.0
    group_cards = [
        {
            "title": k,
            "share_pct": v / total_groups * 100.0,
            "explanation": {
                "Ритм surprisal": "Блок признаков, связанных с динамикой неожиданности текста.",
                "Стабилизаторы текста": "Дополнительные текстовые признаки, которые делают решение устойчивее.",
                "Базовый детектор": "Сигнал исходного детектора, усиленный DivEye.",
            }[k]
        }
        for k, v in group_scores.items()
        if v > 0
    ]

    if is_ai:
        summary_text = (
            "Улучшенный DivEye показал, что текст ближе к ИИ: вклад внесли и ритм surprisal, "
            "и дополнительные стабилизирующие признаки."
        )
    else:
        summary_text = (
            "Улучшенный DivEye показал, что текст ближе к человеческому: ритм surprisal и "
            "дополнительные признаки не дают сильного сигнала в пользу ИИ."
        )

    s = np.asarray(surprisal_seq, dtype=np.float32)
    x = np.arange(1, len(s) + 1)

    if len(s) >= 5:
        window = max(5, min(25, len(s) // 10))
        smooth = pd.Series(s).rolling(window=window, min_periods=1).mean().to_numpy()
    else:
        smooth = s.copy()

    plt.figure(figsize=(20, 9))
    plt.plot(x, s, linewidth=1.8, alpha=0.35, label="Surprisal по токенам")
    plt.plot(x, smooth, linewidth=3.5, label="Сглаженная кривая surprisal")
    plt.title("Ритм неожиданности текста по методу DivEye", fontsize=26, loc="left")
    plt.xlabel("Позиция токена в тексте", fontsize=20)
    plt.ylabel("Неожиданность токена (surprisal)", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(fontsize=16)

    diveye_img_path = "static/diveye_analysis_image.png"
    plt.tight_layout()
    plt.savefig(diveye_img_path, dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "verdict_text": verdict_text,
        "confidence_pct": confidence_pct,
        "description_text": description_text,
        "img_path": diveye_img_path,
        "group_cards": group_cards,
        "top_signals": top_signals,
        "summary_text": summary_text
    }



def extended_analysis_sae_deepseek(text, detector):
    try:
        if detector is None or not getattr(detector, "available", False):
            return {"error": "SAE/DeepSeek/XGB детектор недоступен."}

        if not text or not text.strip():
            return {"error": "Пустой текст."}

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import xgboost as xgb

        # 1. Токенизация и проход через DeepSeek
        X, sae_latents, batch = detector._extract_features([text], return_latents=True)

        # 2. Предсказание и покомпонентные вклады XGBoost
        booster = detector.clf.get_booster()
        dmatrix = xgb.DMatrix(X)

        contribs = booster.predict(dmatrix, pred_contribs=True)  # [1, F+1]
        contribs = contribs[0]

        feature_contribs = contribs[:-1]
        bias_term = float(contribs[-1])

        prob_ai = float(detector.clf.predict_proba(X)[0, 1])
        prediction = "ИИ-ГЕНЕРИРОВАННЫЙ" if prob_ai >= 0.5 else "Человеческий"

        pooled_vec = X[0]

        top_abs_idx = np.argsort(np.abs(feature_contribs))[::-1][:10]

        top_df = pd.DataFrame({
            "feature_idx": top_abs_idx,
            "contribution": feature_contribs[top_abs_idx],
            "abs_contribution": np.abs(feature_contribs[top_abs_idx]),
            "activation": pooled_vec[top_abs_idx],
        }).sort_values("abs_contribution", ascending=True)

        pos_idx = np.where(feature_contribs > 0)[0]
        neg_idx = np.where(feature_contribs < 0)[0]

        top_ai_idx = pos_idx[np.argsort(feature_contribs[pos_idx])[::-1][:5]] if len(pos_idx) > 0 else np.array([], dtype=int)
        top_human_idx = neg_idx[np.argsort(feature_contribs[neg_idx])[:3]] if len(neg_idx) > 0 else np.array([], dtype=int)

        top_ai_features = []
        for idx in top_ai_idx:
            top_ai_features.append({
                "feature_idx": int(idx),
                "contribution": float(feature_contribs[idx]),
                "activation": float(pooled_vec[idx]),
            })

        top_human_features = []
        for idx in top_human_idx:
            top_human_features.append({
                "feature_idx": int(idx),
                "contribution": float(feature_contribs[idx]),
                "activation": float(pooled_vec[idx]),
            })

        plt.figure(figsize=(12, 5))
        colors = ["#c0392b" if v > 0 else "#2980b9" for v in top_df["contribution"]]
        plt.barh(
            [f"SAE #{int(i)}" for i in top_df["feature_idx"]],
            top_df["contribution"],
            color=colors
        )
        plt.axvline(0, color="black", linewidth=1)
        plt.title("Топ-10 SAE-признаков по вкладу в решение")
        plt.xlabel("Вклад в решение XGBoost")
        plt.ylabel("SAE-признак")
        plt.tight_layout()

        barplot_path = "static/sae_analysis_barplot.png"
        plt.savefig(barplot_path, bbox_inches="tight")
        plt.close()

        if len(top_abs_idx) > 0:
            top_feature_idx = int(top_abs_idx[0])

            token_latents = sae_latents[0, :, top_feature_idx].detach().float().cpu().numpy()
            token_ids = batch["input_ids"][0].detach().cpu().tolist()
            tokens = detector.tokenizer.convert_ids_to_tokens(token_ids)

            attn_mask = batch["attention_mask"][0].detach().cpu().numpy().astype(bool)
            token_latents = token_latents[attn_mask]
            tokens = [tok for tok, keep in zip(tokens, attn_mask) if keep]

            x = np.arange(len(token_latents))

            plt.figure(figsize=(13, 4.5))
            plt.plot(x, token_latents, linewidth=1.8)
            plt.title(f"Активация ключевого SAE-признака #{top_feature_idx} по токенам")
            plt.xlabel("Позиция токена в тексте")
            plt.ylabel("Активация признака")
            plt.tight_layout()

            activation_plot_path = "static/sae_top_feature_trace.png"
            plt.savefig(activation_plot_path, bbox_inches="tight")
            plt.close()

            token_preview = []
            for i, (tok, val) in enumerate(zip(tokens[:80], token_latents[:80])):
                token_preview.append({
                    "pos": i,
                    "token": tok,
                    "activation": float(val),
                })
        else:
            top_feature_idx = None
            activation_plot_path = None
            token_preview = []

        total_pos = float(np.sum(feature_contribs[feature_contribs > 0])) if np.any(feature_contribs > 0) else 0.0
        total_neg = float(np.sum(np.abs(feature_contribs[feature_contribs < 0]))) if np.any(feature_contribs < 0) else 0.0
        total_abs = float(np.sum(np.abs(feature_contribs))) if np.sum(np.abs(feature_contribs)) > 0 else 1.0

        ai_signal_pct = round(100.0 * total_pos / total_abs, 1)
        human_signal_pct = round(100.0 * total_neg / total_abs, 1)
        dominance_pct = round(100.0 * np.sum(np.abs(feature_contribs[top_abs_idx[:3]])) / total_abs, 1)

        explanation = {
            "prediction": prediction,
            "ai_score": round(prob_ai, 4),
            "bias_term": round(bias_term, 4),
            "summary": {
                "ai_signal_pct": ai_signal_pct,
                "human_signal_pct": human_signal_pct,
                "dominance_pct": dominance_pct,
            },
            "top_ai_features": top_ai_features,
            "top_human_features": top_human_features,
            "top_feature_idx": top_feature_idx,
            "barplot_path": "/" + barplot_path,
            "activation_plot_path": ("/" + activation_plot_path) if activation_plot_path else None,
            "token_preview": token_preview,
            "layer": detector.layer,
            "hookpoint_name": detector.hookpoint_name,
            "model_name": detector.model_name,
        }
        return explanation

    except Exception as e:
        return {"error": f"Ошибка анализа SAE/DeepSeek: {str(e)}"}

# Flask App Setup
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '')
    file = request.files.get('file')

    if file:
        text = file.read().decode('utf-8')

    if text.strip() == "":
        return render_template('index.html', error="Пожалуйста, введите текст или загрузите файл.")

    with open("temp_text.txt", "w", encoding="utf-8") as f:
        f.write(text)

    result = combined_detector.predict(text)
    return render_template('index.html', result=result, text_value=text)

@app.route('/extended_analysis_page', methods=['POST'])
def extended_analysis_page():
    if not os.path.exists("temp_text.txt"):
        return render_template(
            'extended_analysis_page.html',
            dep=None,
            diveye=None,
            sae=None,
            error="Не удалось выполнить расширенный анализ: исходный текст не найден."
        )

    with open("temp_text.txt", "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return render_template(
            'extended_analysis_page.html',
            dep=None,
            diveye=None,
            sae=None,
            error="Не удалось выполнить расширенный анализ: текст пуст."
        )

    diveye_ai_prob = float(request.form.get("diveye_ai_prob", 0.5))

    dep = extended_analysis(text, detector=dependency_model)
    diveye = extended_analysis_diveye(text, diveye_model, diveye_ai_prob)
    sae = extended_analysis_sae_deepseek(text, sae_xgb_model)

    return render_template(
        'extended_analysis_page.html',
        dep=dep,
        diveye=diveye,
        sae=sae,
        error=None
    )

if __name__ == '__main__':
    # Initialize the models
    dependency_model = DependencyAIDetector('dependency_vectorizer.pkl', 'dependency_model.pkl')
    diveye_model = RussianAIDetector(xgb_path="diveye_llmtrace_ru_xgb.pkl")
    sae_xgb_model = SAEDeepSeekXGBDetector(
        deepseek_root='./deepseek',
        config_path='./deepseek/artifacts/run_config.json',
        xgb_path=None,
        model_path='./deepseek/DeepSeek-R1-Distill-Qwen-1.5B',
        sae_path=None,
    )
    combined_detector = CombinedAIDetector(dependency_model, diveye_model, sae_xgb_model)
    

    app.run(debug=False, use_reloader=False, port=5001)
