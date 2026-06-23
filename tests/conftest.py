import pytest
from lexausearch.models import Chunk

AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"

PRIVACY_ACT_XML = f"""\
<?xml version='1.0' encoding='UTF-8'?>
<akomaNtoso xmlns="{AKN_NS}">
  <act name="act">
    <meta>
      <identification source="#lex-au">
        <FRBRWork>
          <FRBRthis value="/akn/au/act/1988/119/!main"/>
          <FRBRuri value="/akn/au/act/1988/119"/>
          <FRBRdate date="1988" name="Generation"/>
          <FRBRauthor href="#parliament"/>
        </FRBRWork>
        <FRBRExpression>
          <FRBRthis value="/akn/au/act/1988/119/eng@2026-01-01/!main"/>
          <FRBRuri value="/akn/au/act/1988/119/eng@2026-01-01"/>
          <FRBRdate date="2026-01-01" name="Generation"/>
          <FRBRauthor href="#parliament"/>
          <FRBRlanguage language="eng"/>
        </FRBRExpression>
        <FRBRManifestation>
          <FRBRthis value="/akn/au/act/1988/119/eng@2026-01-01/!main.akn"/>
          <FRBRuri value="/akn/au/act/1988/119/eng@2026-01-01/!main.akn"/>
          <FRBRdate date="2026-06-19" name="Generation"/>
          <FRBRauthor href="#lex-au"/>
        </FRBRManifestation>
      </identification>
    </meta>
    <body>
      <section eId="sec-3">
        <num>3</num>
        <heading>Interpretation</heading>
        <content>
          <p>In this Act personal information means information about an identified individual.</p>
        </content>
      </section>
      <section eId="sec-6">
        <num>6</num>
        <heading>Meaning of personal information</heading>
        <content>
          <p>Personal information means information or an opinion about an identified individual.</p>
          <p>The information need not be recorded in material form.</p>
        </content>
      </section>
    </body>
  </act>
</akomaNtoso>"""


@pytest.fixture(scope="module")
def privacy_chunks() -> list[Chunk]:
    return [
        Chunk(
            act_name="Privacy Act 1988",
            frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
            eid="sec-3",
            provision_num="3",
            provision_type="section",
            heading="Interpretation",
            text="3 Interpretation In this Act personal information means information about an identified individual.",
            refs=[],
        ),
        Chunk(
            act_name="Privacy Act 1988",
            frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
            eid="sec-6",
            provision_num="6",
            provision_type="section",
            heading="Meaning of personal information",
            text="6 Meaning of personal information Personal information means information or an opinion about an identified individual. The information need not be recorded in material form.",
            refs=[],
        ),
    ]
