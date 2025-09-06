from sqlalchemy.orm import Session
from sqlalchemy import func
import models
import schemas

def create_assessment(db: Session, assessment: schemas.AssessmentCreate):
    db_assessment = models.UserAssessment(
        name=assessment.name,
        location=assessment.location,
        dwellers=assessment.dwellers,
        roof_area=assessment.roof_area,
        open_space=assessment.open_space,
        roof_type=assessment.roof_type
    )
    db.add(db_assessment)
    db.commit()
    db.refresh(db_assessment)
    return db_assessment

def get_assessment(db: Session, assessment_id: int):
    return db.query(models.UserAssessment).filter(models.UserAssessment.id == assessment_id).first()

def get_all_assessments(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.UserAssessment).offset(skip).limit(limit).all()

def update_assessment_results(db: Session, assessment_id: int, results: dict):
    db_assessment = db.query(models.UserAssessment).filter(models.UserAssessment.id == assessment_id).first()
    if db_assessment:
        # Update only the fields that are provided in results
        for key, value in results.items():
            if hasattr(db_assessment, key):
                setattr(db_assessment, key, value)
        db.commit()
        db.refresh(db_assessment)
    return db_assessment

def delete_assessment(db: Session, assessment_id: int):
    db_assessment = db.query(models.UserAssessment).filter(models.UserAssessment.id == assessment_id).first()
    if db_assessment:
        db.delete(db_assessment)
        db.commit()
        return True
    return False